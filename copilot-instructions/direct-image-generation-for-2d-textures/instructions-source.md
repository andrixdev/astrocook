Script astroviz-csharp-script.txt is a C# script for Unity that parses text data into 2D Textures (images) containing the same data but encoded in the image, with specific formatting.
It currently lives within a Unity project that transforms the current astrocook project outputs into visualizations.

This project (astrocook) currently generates, from various data formats:
- 3D textures, ready to be processed by Unity
- text files, that Unity parses to 2D textures via the astroviz-csharp-script.txt script

The goal is to migrate it to the current astrocook project.

Propose suggestions for a new script that goes along particles_textufy.py in the "astrocutlerty" folder that can go futher than generating text files by also generating ready-to-use 2D Texture images.
I feel that keeping the textual data file is still useful for visual data debugging.
Propose a new clean process that outputs both txt and img files.
Current Unity script also generated folders for extra data dimensions. This should now be performed in Python.

Also list all astroviz-csharp-script.txt does so I don't forget anything in the next prompt.

Keep code as concise as possible
Minimize credit consumption
