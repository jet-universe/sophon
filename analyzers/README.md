## Delphes analyzer (C++ macro) for Sophon model inference

In the following example, we take a Delphes ROOT file `events_delphes_example.root` as input, run an analyzer macro `analyze.C` to infer the Sophon model, and output both scores and latent space features to a new ROOT file `out.root`.

This example works on EL9 machines.

```bash
# Setup environment

source /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-el9-gcc13-opt/setup.sh
export ROOT_INCLUDE_PATH=$ROOT_INCLUDE_PATH:/cvmfs/sft.cern.ch/lcg/releases/delphes/3.5.1pre09-9fe9c/x86_64-el9-gcc13-opt/include:/cvmfs/sft.cern.ch/lcg/releases/onnxruntime/1.15.1-8b3a0/x86_64-el9-gcc13-opt/include/core/session

# Compile and run macro

root -b -q 'analyze.C++("events_delphes_example.root", "out.root", "../models/JetClassII_Sophon/model.onnx")'
```

**Note:**

To ensure that the Sophon model achieves the expected performance, it is highly recommended that the Delphes file is produced from the **`delphes_card_CMS_JetClassII`** card series provided in the [`jetclass2_generation`](https://github.com/jet-universe/jetclass2_generation) repository.

This card is based on the CMS detector configuration, applying an additional track smearing, emulating the PU effect with $<\mu>$=50 and mitigating PU with the PUPPI algorithm.
