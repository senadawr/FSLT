# Architecture

The browser owns camera access and MediaPipe Holistic processing. Only the
versioned landmark frame contract is sent over `/ws/inference`; raw video is
not uploaded or stored. The FastAPI service validates frames, keeps a bounded
buffer, and loads a configured model once at startup.

Isolated recognition and continuous recognition are separate model paths.
Continuous recognition will use a temporal encoder and CTC once uninterrupted
FSL signing data with gloss annotations is available.
