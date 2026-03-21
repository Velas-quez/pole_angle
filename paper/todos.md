- Posicionamento na literatura fraco — Não fica claro como o trabalho se relaciona com pesquisas anteriores em estimação de estado tátil. Faltam citações cruciais (inclusive do próprio EveTac na seção de Background).
- ~Statistical Multi-Quad encoding pouco explicado — Sendo uma contribuição central, a descrição é insuficiente. Os revisores precisaram inferir detalhes. Sugerem introduzi-lo já na introdução ou em um overview.~  **(Pepe)**
- ~Hu moments precisam de mais explicação — Atualmente parece superficial ("hand-wavy"). Um revisor sugere adicionar uma figura.~ **(PEPE)**
- Baselines adicionais sugeridas — Um modelo treinado só com dados táteis (sem estado do robô) ajudaria a contextualizar a contribuição do sensor. Um revisor também sugere comparação com loss Euclidiana vs. arccos para justificar a escolha geométrica. ()
- Definição de "support" — Usado sem definição clara ao longo do texto. (**Pepe**)
- A Seção 5.2 ficaria mais bem posicionada no Capítulo 2

CONCLUIDOS

- Tabela 1 não é referenciada no texto **(Velas)**
- Melhor modelo não está em negrito nas tabelas **(Velas)**
- Adicionar figura de exemplo de um mapa de pressão tátil **(Velas)**
- - TCN usado antes de ser definido por extenso **(Velas)**
- Distinção entre estimação de orientação e balanceamento — O paper foca em prever orientação, mas o título e motivação falam em balancear o polo. Essa distinção precisa ser explicitada claramente. **(Velas)**
- Final performance - explicar melhor que o erro final é ou nn é suficiente para treinar o modelo **(Velas)**
- Figura 2 pouco explicada — Não está claro o que o leitor deve concluir da comparação entre os dois membros da equipe.**(Velas)**
- Alta variância nos resultados — Em vários casos, o desvio padrão é comparável à média. Os revisores questionam se as diferenças entre modelos são estatisticamente significativas.**(Velas)**