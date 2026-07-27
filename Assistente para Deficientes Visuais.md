# Modelagem Funcional Resumo

Este documento apresenta o resumo da modelagem funcional para a nova arquitetura do sistema BlindDistance, com um sistema de **Visão Estéreo** baseado em duas webcams convencionais.

---

## 1. Requisitos Técnicos

### Requisitos de Hardware
* **Câmeras:** 2 webcams USB comuns.
* **Montagem Mecânica:** uporte rígido garantindo que as duas câmeras mantenham uma distância fixa entre si (baseline de 6 a 7 cm).
* **Ferramenta de Calibração:** Um padrão impresso de tabuleiro de xadrez (8x6 cantos, quadrados de 30mm).
* **Processamento:** Computador com processador multi-core.
* **Interface de Saída:** Alto-falantes ou fones de ouvido para feedback de áudio.

### Requisitos de Software
* **Base:** Linguagem Python (3.12+).
* **Visão Computacional:** Biblioteca `opencv` para captura de imagens, calibração estéreo, retificação, e cálculo do mapa de disparidade.
* **Inteligência Artificial:** Modelo YOLOv8 Nano (`ultralytics`) para reconhecimento rápido de objetos em tempo real.
* **Áudio:** Bibliotecas `pyttsx3` para conversão de texto em fala (TTS) offline e `pygame` para geração de bipes de alerta contínuos.

---

## 2. Especificações

O sistema opera triangulando a diferença de perspectiva entre as duas câmeras (disparidade) para calcular a distância dos objetos. 

* **Alcance Útil Operacional Esperado:** 0.5 metros a 3.0 metros.
* **Performance Alvo:** Mínimo de 15 FPS de processamento ponta a ponta.
* **Resolução Operacional:** Captura e processamento otimizados para 640x480 pixels.

---

## 3. Funções e Funcionalidades do Sistema

O sistema é comporto por um fluxo contínuo que executa as seguintes funções principais:

1. **Captura Estéreo Sincronizada:** Leitura simultânea dos *frames* da câmera esquerda e direita.
2. **Retificação de Imagem:** Calibração para alinhar horizontalmente as imagens capturadas (linhas epipolares).
3. **Cálculo de Profundidade (StereoSGBM):** Geração de Mapa de Disparidade para cálculo de profundidade.
4. **Alerta de Colisão / Proximidade (Grade):** Divisão virtual do mapa de profundidade para monitorar barreiras iminentes (objetos a menos de 1 metro). Se detectados, o sistema aciona *bipes sonoros* urgentes.
5. **Detecção Inteligente de Objetos (YOLOv8):** A inteligência artificial analisa a imagem colorida para entender *o que* é o obstáculo. O sistema cruza a posição do objeto com o mapa de profundidade e anuncia sua distância via *voz sintetizada*.
6. **Detecção de Riscos no Piso (Quedas/Degraus):** Análise constante do terço inferior da visão da câmera. Caso a profundidade aumente de forma repentina (indicando um buraco ou escada descendente), um alerta verbal de emergência é acionado.

---

## 4. Diagramas de Blocos de Implementação (Descritivo)

![[Pasted image 20260714205332.png]]

---

## 5. Método de Calibração

O sistema de visão estéreo requer um **método de calibração manual**.

1. **Ferramenta:** Utiliza-se um script dedicado (`stereo_calibration.py`).
2. **Coleta de Dados:** O usuário posiciona um padrão de tabuleiro de xadrez impresso em frente às câmeras e o movimenta. O sistema captura entre 15 e 30 fotos em que o tabuleiro seja perfeitamente visível em ambas as lentes simultaneamente.
3. **Cálculo:** O OpenCV reconhece os cantos do tabuleiro e calcula os coeficientes de distorção das lentes (calibração intrínseca) e a rotação/translação exata entre as duas câmeras (calibração extrínseca).
4. **Armazenamento:** Os parâmetros são salvos em um arquivo `.xml`. Quando o sistema principal (`main.py`) rodar, ele carregará esses dados para corrigir a imagem e calcular a profundidade com precisão matemática.

---

## 6. Método de Avaliação Funcional

Para garantir que o software atinge os requisitos assistivos, o projeto passará pelos seguintes métodos de validação:

1. **Avaliação Qualitativa da Calibração:** Verificação da métrica e Visualização do teste de retificação para confirmar o alinhamento horizontal (epipolar).
2. **Validação Quantitativa da Profundidade:** Com o sistema ativo, posicionar obstáculos a distâncias conhecidas (0.5m, 1m, 2m, medidas com fita métrica) e comparar o resultado gerado pelo algoritmo.
3. **Validação Comportamental de Segurança:**
   * Caminhar de encontro a uma parede plana para certificar que o alerta de proximidade  dispara antes da colisão (x < 1.0 metro).
   * Apontar a câmera levemente para baixo em direção a um lance de escadas descendente para verificar se o sistema detecta a quebra do piso e aciona o alerta de voz de queda.
1. **Validação de Performance:** Medição do *Framerate* (FPS) médio do sistema rodando todas as camadas simultaneamente (IA, Disparidade e Áudio), que deve permanecer igual ou superior a 15 FPS para não causar atraso nas respostas ao usuário.

---
Nota de Autoria: O conteúdo e as ideias expressas neste documento são de autoria humana original. Ferramentas de Inteligência Artificial foram empregadas de forma auxiliar, exclusivamente para a formatação e organização visual do texto._