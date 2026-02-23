#!/usr/bin/env python
import os

import datetime

import dv_processing as dv
import cv2 as cv
import argparse
import numpy as np
import time
import pickle
import copy
import torch
import scipy
import rospy
from rospy.numpy_msg import numpy_msg
from ebs_custom_msgs.msg import EBStampedFloats
from ebs_custom_msgs.msg import EBFloats
from ebs_custom_msgs.msg import EBInt16
from ebs_custom_msgs.msg import EBInt8
from ebs_custom_msgs.msg import EBTimestamp
from ebs_custom_msgs.msg import EvetacMsg

# from ebs_custom_msgs.msg import EBDims
# from ebs_custom_msgs.msg import EBCOO
from sys import exit
from std_srvs.srv import Trigger, TriggerResponse
from sensor_msgs.msg import Image as SensorImage


from ebvision.utils import GenerateImageOnnx, DotTrackOnnx

from threading import Thread

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)
torch.no_grad

# parser = argparse.ArgumentParser(description='Show a preview of an iniVation event camera input.')
# parser.add_argument('--path', type=str, default='',
#                     help='path where the calibration should be stored')
# parser.add_argument('--name', type=str, default='new_calibration1_12_0.pkl',
#                     help='name of the calibration')
# parser.add_argument('--recording', type=str, default='',
#                     help='path where the calibration should be stored')
# parser.add_argument('--pub_raw_events', type=bool, default=False,
#                     help='whether to publish raw events or not')
# parser.add_argument('--cam_name', type=str, default="",
#                     help='provide name of the camera')
# parser.add_argument('--cam_id', type=str, default="0",
#                     help='provide identifier of the camera')
# parser.add_argument('--inivation_preview', type=bool, default=False,
#                     help='whether to show the preview or not')
# parser.add_argument('--show_neighbors', type=bool, default=False,
#                     help='whether to show the neighbors in the inivation preview or not. Only has an effect if inivation_preview is set to True')
# parser.add_argument('--use_gpu', type=bool, default=False,
#                     help='whether to use the gpu or not')
# parser.add_argument('--crop', type=int, default=0,
#                     help='whether to crop the image for tracking or not, specify in negative pixels, cropped at the bottom')

# args = parser.parse_args()

# TODO what namespace for the params?
# print(os.getcwd())
# path = rospy.get_param("/ebcam_driver/path", "/catkin_ws/src/ebvision/calibration/calibrations/")
path = rospy.get_param(
    "/ebcam_driver/path",
    "cartkin_ws/src/ebvision/ebcam_driver/src/ebvision/calibration/calibrations",
    #  "/catkin_ws/src/ebvision/calibration/calibrations/"
)
name = rospy.get_param("/ebcam_driver/name", "new_calibration_20_02_2024.pkl")
recording = rospy.get_param("/ebcam_driver/recording", "")
pub_raw_events = rospy.get_param("/ebcam_driver/pub_raw_events", False)
cam_name = rospy.get_param("/ebcam_driver/cam_name", "")
cam_id = str(rospy.get_param("/ebcam_driver/cam_id", 0))
inivation_preview = rospy.get_param("/ebcam_driver/inivation_preview", False)
show_neighbors = rospy.get_param("/ebcam_driver/show_neighbors", False)
use_gpu = rospy.get_param("/ebcam_driver/use_gpu", False)
crop = rospy.get_param("/ebcam_driver/crop", 0)


downsample_factor = 8
# define by how much the sensor is slowed down
downsample_factor = int(np.clip(downsample_factor, 1, 10))

print(
    "Sensor slowed down by factor: ",
    downsample_factor,
    " therefore new operating frequency: ",
    1000 / downsample_factor,
    " Hz",
)

# potentially print all the available connected devices
print(dv.io.discoverDevices())

if recording != "":
    # use recording
    camera = dv.io.MonoCameraRecording(recording)
else:
    # potentially print all the available connected devices
    print(dv.io.discoverDevices())
    # Open the camera
    print("Using camera: ", cam_name)
    camera = dv.io.CameraCapture(cam_name)

# print the internal integration time of the event-based camera, default set to 10000 us
# print(camera.deviceConfigGet(CAER_HOST_CONFIG_PACKETS, CAER_HOST_CONFIG_PACKETS_MAX_CONTAINER_INTERVAL))
print(camera.deviceConfigGet(-3, 1))
# set the internal integration time of the event-based camera to 500 us to not create an unnecessary delay
# until events are sent to the computer
# camera.deviceConfigSet(CAER_HOST_CONFIG_PACKETS, CAER_HOST_CONFIG_PACKETS_MAX_CONTAINER_INTERVAL, 1000)
camera.deviceConfigSet(-3, 1, 1000)
# # eventually again print integration time again to verify it was set correctly
# print(camera.deviceConfigGet(-3, 1))


class CameraProcessor:
    def __init__(self, camera):
        self.pub_evetac_msg = rospy.Publisher(
            "/evetac_msg" + cam_id, EvetacMsg, queue_size=10
        )

        rospy.init_node("talker", anonymous=True)

        self.pub_timestamp = rospy.Publisher(
            "timestamp" + cam_id, EBTimestamp, queue_size=250
        )

        # Create an event slicer, this will only be used events only camera
        self.slicer = dv.EventStreamSlicer()
        self.slicer.doEveryTimeIntervalAndReturnInterval(
            datetime.timedelta(microseconds=1000 * downsample_factor),
            self.callback_events,
        )
        self.use_gpu = use_gpu
        self.camera = camera
        self.image_generator = GenerateImageOnnx(
            self.camera, gpu=self.use_gpu, generate_model_anew=True
        )
        self.pic = self.image_generator.get_zero_img()
        self.tracker = DotTrackOnnx(path + name, gpu=self.use_gpu)
        self.dot_locations = self.tracker.get_dot_locations()

        if crop >= 0:
            print("No cropping")
            self.use_crop = False
        else:
            print("Cropping")
            self.use_crop = True
            self.crop = int(camera.getEventResolution()[0] + crop)

        # Initialize visualizer instance which generates event data preview
        self.visualizer = dv.visualization.EventVisualizer(camera.getEventResolution())
        # dot regularization factor:

        # create a filter chain to do some filtering
        self.use_lighter_prefilter = False
        self.filter_chain = dv.EventFilterChain()
        self.filter_chain.addFilter(
            dv.noise.BackgroundActivityNoiseFilter(camera.getEventResolution())
        )
        self.filter_chain_raw = dv.EventFilterChain()
        self.filter_chain_raw.addFilter(
            dv.noise.BackgroundActivityNoiseFilter(
                camera.getEventResolution(), datetime.timedelta(microseconds=100000)
            )
        )
        # self.filter_chain_raw.addFilter(dv.noise.BackgroundActivityNoiseFilter(camera.getEventResolution()))
        # self.filter_chain.addFilter(dv.noise.BackgroundActivityNoiseFilter(camera.getEventResolution()))
        # create variables for the timestep (needed for visualization), as well as the coordinate for the tracking

        self.current_time = 0
        self.counter = 0
        self.fail_counter = 0
        self.iter_counter = 0
        self.global_time = 0

        self.show_inivation_preview = inivation_preview
        self.show_neighbors = show_neighbors

        self.copy = dv.EventStore()

        # Create the preview window
        # cv.namedWindow("Preview", cv.WINDOW_NORMAL)

        self.tracking_active = True
        # Create the rosservice to trigger storing the calibration
        self.stop_track_service = rospy.Service(
            "/stop_keypoint_track" + cam_id, Trigger, self.stop_keypoint_track
        )
        self.start_track_service = rospy.Service(
            "/start_keypoint_track" + cam_id, Trigger, self.start_keypoint_track
        )
        self.store_current_calibraation = rospy.Service(
            "/store_current_dot_calibration" + cam_id,
            Trigger,
            self.store_current_calibration,
        )
        self.reinit_tracker_srv = rospy.Service(
            "/reinit_tracker_srv" + cam_id, Trigger, self.reinit_tracker
        )

    def stop_keypoint_track(self, request):
        self.tracking_active = False
        return TriggerResponse(
            success=True,
            message="Stopped tracking - tracking status now: "
            + str(self.tracking_active),
        )

    def start_keypoint_track(self, request):
        self.tracking_active = True
        return TriggerResponse(
            success=True,
            message="Stopped tracking - tracking status now: "
            + str(self.tracking_active),
        )

    def store_current_calibration(self, request):
        # Function to store the current calibration (i.e., location of the dots)
        start = time.time()
        self.tracker.store_current_calibration()
        print("Stored current calibration. This took: ", time.time() - start)

    def reinit_tracker(self, request):
        # function to reinit the tracker (i.e., reinitialize the reference positions to the latest calibration)
        start = time.time()
        self.tracker.reinitialize()
        print("Reinitialized the dot tracking. This took: ", time.time() - start)

    def create_cross(self, event_slice, pos_x, pos_y, size=3, curr_time=0):
        # note last argument must be the highest time of the event slice!
        event_slice.push_back(curr_time, pos_x, pos_y, False)
        # print ("highest time: ", event_slice.getHighestTime())
        for i in range(int(size)):
            event_slice.push_back(curr_time, pos_x - i, pos_y, False)
            event_slice.push_back(curr_time, pos_x + i, pos_y, False)
            event_slice.push_back(curr_time, pos_x, pos_y + i, False)
            event_slice.push_back(curr_time, pos_x, pos_y - i, False)
        return event_slice

    def draw_line(self, event_slice, start_x, start_y, end_x, end_y, curr_time):
        dx = end_x - start_x
        dy = end_y - start_y
        spacing = np.linspace(0.1, 0.25, num=10)
        dx = dx * spacing + start_x
        dy = dy * spacing + start_y
        for i in range(spacing.shape[0]):
            event_slice.push_back(curr_time, int(dx[i]), int(dy[i]), False)
        return event_slice

    def visualize_neighbors(
        self, event_slice, x_coords, y_coords, pairwise_dists_mask, curr_time
    ):
        for i in range(x_coords.shape[0]):
            event_slice.push_back(curr_time, int(x_coords[i]), int(y_coords[i]), False)
            nonzero_idcs = torch.nonzero(pairwise_dists_mask[i, :])
            for j in range(nonzero_idcs.size()[0]):  # this retrieves the neighbors
                event_slice = self.draw_line(
                    event_slice,
                    int(x_coords[i]),
                    int(y_coords[i]),
                    int(x_coords[nonzero_idcs[j]]),
                    int(y_coords[nonzero_idcs[j]]),
                    curr_time,
                )
        return event_slice

    def inivation_preview(self, event_slice):
        # increment the counter
        self.counter += 1
        self.copy.add(event_slice)

        # this criteria makes sure that we only show the visualization at a slower frequency
        if (self.counter % 33) == 0:
            event_slice = self.copy
            self.current_time = event_slice.getHighestTime()
            if self.show_neighbors:
                # get the relevant info from the tracker:

                # eventually visualize neighbors:
                event_slice = self.visualize_neighbors(
                    event_slice,
                    self.dot_locations[:, 1],
                    self.dot_locations[:, 0],
                    self.tracker.pairwise_dists_mask,
                    self.current_time,
                )
                # TODO if (False): # eventually visualize the circle center points:
                for kk in range(np.shape(self.calib_center)[0]):
                    # # eventually also visualize their radius:
                    event_slice = self.create_cross(
                        event_slice,
                        int(self.calib_center[kk, 1]),
                        int(self.calib_center[kk, 0]),
                        self.calib_radius[kk] - 12.5,
                        self.current_time,
                    )
                    # event_slice = self.create_cross(event_slice, int(self.calib_center[kk, 1]),
                    #                                 int(self.calib_center[kk, 0]), 3, self.current_time)

            # cv.imshow("Preview", self.visualizer.generateImage(event_slice))
            # cv.waitKey(1)
            self.copy = dv.EventStore()

    def callback_events(self, event_interval, event_slice):

        start = time.time()
        timestamp = rospy.Time.now()
        self.iter_counter += 1
        # reset iter_counter when both modulos would be active anyways
        if self.iter_counter == 256 * 1000:
            self.iter_counter = 0
            # also reset the fail counter then:
            self.fail_counter = 0

        if self.use_lighter_prefilter:
            self.filter_chain_raw.accept(event_slice)
            event_slice = self.filter_chain_raw.generateEvents()

            events_numpy_unfiltered = event_slice.numpy()
            events_x_unfiltered = (
                torch.from_numpy(events_numpy_unfiltered["x"]).cuda().long()
            )
            events_y_unfiltered = (
                torch.from_numpy(events_numpy_unfiltered["y"]).cuda().long()
            )
            events_polarity_unfiltered = (
                torch.from_numpy(events_numpy_unfiltered["polarity"])
                .cuda()
                .type(torch.int8)
            )
            events_polarity_unfiltered[events_polarity_unfiltered == 0] = -1

        # # shoot events through the filter
        # self.filter_chain.accept(event_slice)
        # event_slice = self.filter_chain.generateEvents()

        self.filter_chain_raw.accept(event_slice)
        event_slice = self.filter_chain_raw.generateEvents()

        # convert events to numpy
        events_numpy = event_slice.numpy()

        if len(events_numpy["x"]) > 0:
            # coords = list(map(EBCOO, events_numpy['x'], events_numpy['y'], events_numpy['polarity']))
            self.pic = self.image_generator.generate_image(
                events_numpy["x"], events_numpy["y"], events_numpy["polarity"]
            )

            # TODO
            if self.tracking_active:
                # start = time.time()
                if not (self.use_crop):
                    self.dot_locations = self.tracker.track(
                        events_numpy["x"], events_numpy["y"]
                    )
                    pass
                else:
                    decider = events_numpy["x"] < self.crop
                    if np.any(decider):
                        self.dot_locations = self.tracker.track(
                            events_numpy["x"][decider], events_numpy["y"][decider]
                        )
                        pass
                # print ("tracking took: ", time.time()-start)
        else:
            # pass
            self.pic[:] = 127
            # coords = []

        num_eventos = (self.pic != 127).sum()
        print(f"Total de pontos ativos: {num_eventos}")
        print(f"Tamanho da imagem: {self.pic.shape}")
        print(f"Value of active pixels: {self.pic[self.pic != 127]}")

        if not rospy.is_shutdown():
            local_counter = np.int8((self.iter_counter % 256) - 128)

            msg = EvetacMsg()
            msg.header.stamp = timestamp

            msg.image = SensorImage()
            msg.image.height = 640
            msg.image.width = 480
            msg.image.encoding = "mono8"
            msg.image.is_bigendian = False
            msg.image.step = 480
            msg.image.data = self.pic.tobytes()
            # msg.dims = EBDims(rows = 640,cols = 480)
            # msg.coords = coords

            msg.local_counter = local_counter

            msg.centers = self.dot_locations.reshape(-1, order="F")
            # msg.events_x = events_numpy['x'].tolist()
            # msg.events_y = events_numpy['y'].tolist()
            # msg.events_polarity = events_numpy['polarity'].tolist()

            self.pub_evetac_msg.publish(msg)

        else:
            exit()

        curr_duration = time.time() - start

        if curr_duration > 0.001:
            self.fail_counter += 1

        if (self.iter_counter + 1) % int(1000 / downsample_factor) == 0:
            print(time.time() - self.global_time)
            self.global_time = time.time()
            print("FAILS: ", self.fail_counter)
            print("TOTAL: ", self.iter_counter)

        # TODO
        if self.show_inivation_preview:
            self.inivation_preview(event_slice)


if __name__ == "__main__":
    # create instance of class:
    cam_processor = CameraProcessor(camera)

    # start read loop
    # TODO
    while True:
        # while rospy.spin():
        # Get events
        events = camera.getNextEventBatch()
        # print(events)
        # If no events arrived yet, continue reading
        if events is not None:
            cam_processor.slicer.accept(events)
        else:
            time.sleep(0.00025)
