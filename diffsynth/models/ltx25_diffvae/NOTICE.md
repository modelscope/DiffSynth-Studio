# LTX-2.5 DiffVAE eager port notice

The Python sources in this directory are a dependency-closed, in-tree port of
selected `ltx_core` DiffVAE decoder sources from `Lightricks/LTX-2` revision
`400fd31054597515f47125691032c04b1c3ee24e`.  Original source headers are
retained.  The port deliberately excludes NATTEN, Triton, Blackwell DSL, and
runtime imports from the installed `ltx_core` package.  It uses the upstream
pure-PyTorch eager tiled-SDPA implementation as its neighborhood-attention
backend.
