# Notices and attributions

**gebtaero-jax** is a Python/JAX re-implementation inspired by, and validated against, two projects:

## GEBTAero
<https://github.com/mid2SUPAERO/GEBTAero> — High-aspect-ratio composite wing nonlinear aeroelastic toolbox
by Bertrand Kirsch (French Air Force Academy Research Center / ISAE-Supaero).
Its Python pre/post-processor is licensed GPL-3.0-or-later; gebtaero-jax is therefore distributed under the same
license. The input file `examples/data/goland.dat` is copied from that repository.

The GEBTAero computation core is derived from **GEBT** by Prof. Wenbin Yu
(<https://cdmhub.org/resources/gebt>), distributed under the following terms:

> Copyright (c) 2014, Wenbin Yu. All rights reserved.
>
> Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
> following conditions are met: Redistributions of source code must retain the above copyright notice, this list of
> conditions and the following disclaimer. Redistributions in binary form must reproduce the above copyright notice,
> this list of conditions and the following disclaimer in the documentation and/or other materials provided with the
> distribution. Neither the name of Purdue University nor the names of its contributors may be used to endorse or
> promote products derived from this software without specific prior written permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 'AS IS' AND ANY EXPRESS OR IMPLIED WARRANTIES,
> INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
> DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
> SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
> SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
> WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
> OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Aeroelasticity.jl
<https://github.com/byuflowlab/Aeroelasticity.jl> — A modular multi-fidelity aeroelastic analysis and simulation
framework by Taylor McDonnell (BYU FLOW Lab), MIT license. gebtaero-jax re-implements its coupled-model
formulation, 2D aerodynamic models and typical-section model in JAX (no code is copied; the algorithms follow the
published documentation and source).
