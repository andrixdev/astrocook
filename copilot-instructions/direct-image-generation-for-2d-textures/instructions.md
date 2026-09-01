<context>
You are an expert Python and C# software architect assisting with a pipeline migration from Unity to a standalone Python ecosystem. 

Presently, the `astrocook` project generates 3D textures (ready for Unity consumption) alongside raw text files. A Unity C# script (`copilot-instructions/direct-image-generation-for-2d-textures/astroviz-csharp-script.txt`) subsequently ingests these text files to synthesize 2D Textures (encoded images) and orchestrates specific directory hierarchies to accommodate supplementary data dimensions. Examples of usage of this C# script are given in the C# script `copilot-instructions/direct-image-generation-for-2d-textures/astroviz-csharp-script-usage-examples.txt`
</context>

<objective>
Translocate the 2D texture synthesis and directory scaffolding logic from the Unity C# environment into a novel Python script. This subsequent module shall reside adjacent to `astrocutlery/particles_textufy.py` within the `astrocutlery` directory.
</objective>

<requirements>
1. **Comprehensive Functionality Inventory:** Prior to formulating the architectural proposal, meticulously dissect the provided C# script and enumerate every discrete operation it performs. This exhaustively compiled list will serve as our unassailable blueprint for subsequent iterations.
2. **Dual-Output Paradigm:** Propose a pristine, bifurcated pipeline that simultaneously yields the legacy textual data files (which remain indispensable for visual data diagnostics) and the finalized, ready-to-use 2D texture images.
3. **Dimensionality & Directory Scaffolding:** Transfer the responsibility of generating hierarchical folders for extra data dimensions entirely into this novel Python script.
4. **Algorithmic Parsimony:** Craft the Python implementation with maximum brevity and computational efficiency. Minimize token consumption and processing overhead at all junctures. 
5. **Specific single-texture case:** The current C# script writes files into either
the main folder for single-texture outputs or
a subfolder for multi-texture outputs. Now we want to also export into subfolders even for single-textures (same naming)
</requirements>

<input_data>
The main script is available here: `copilot-instructions/direct-image-generation-for-2d-textures/astroviz-csharp-script.txt`
The examples are available here: `copilot-instructions/direct-image-generation-for-2d-textures/astroviz-csharp-script-usage-examples.txt`
A python script from another mathematics projects that successfully outputs well-formatted Unity 2D Textures is given here: `copilot-instructions/direct-image-generation-for-2d-textures/mathviz-python-script.txt`
</input_data>

<output_directives>
Commence your response with the enumerated inventory of the C# script's functionalities. 
Thereafter, delineate your proposed process architecture. 
Conclude with the commented Python implementation.
</output_directives>