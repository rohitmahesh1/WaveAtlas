#!/usr/bin/env wolframscript
(* Determine the directory of the script *)
dir = DirectoryName@$InputFileName;
If[dir == "", dir = NotebookDirectory[]];

(* Load necessary packages *)
Get[FileNameJoin@{dir, "KymoButler_copy.wl"}];
Get[FileNameJoin@{dir, "KymoButlerPProc.wl"}];

(* Set random seed for deterministic results *)
SeedRandom[1234];

(* Load the default neural networks models *)
models = Quiet[loadDefaultNets[dir]];

(* Get command line arguments *)
args = Rest[$ScriptCommandLine];

(* Parse the output directory, indices, and image file *)
If[Length[args] != 2,
    Print["Usage: wolframscript Run_KymoButler_Save_Image.wls outputDir indices imageFile"];
    Exit[];
];

outputDir = args[[1]];
(*indicesArg = ToExpression[args[[2]]];*)
indicesArg = Null;
imageFile = args[[2]];

(* Create output directory if it does not exist *)
If[!DirectoryQ[outputDir], CreateDirectory[outputDir]];

(* Function to save images *)
saveImages[images_, baseName_, dir_] := Module[{outputFileName, image},
    If[ListQ[images],
        (* Save as a sequence of images *)
        Do[
            image = If[Head[images[[i]]] === Graphics, Rasterize[images[[i]]], images[[i]]];
            If[ImageQ[image],
                outputFileName = FileNameJoin@{dir, baseName <> "_processed_" <> ToString[i] <> ".png"};
                Export[outputFileName, image];
                Print["Saved processed image frame to: ", outputFileName];
            , Print["Item at index ", i, " is not an image: ", images[[i]]]];
        , {i, Length[images]}];
    ,
        (* Save as a single image if images is not a list *)
        image = If[Head[images] === Graphics, Rasterize[images], images];
        If[ImageQ[image],
            outputFileName = FileNameJoin@{dir, baseName <> "_processed.png"};
            Export[outputFileName, image];
            Print["Saved processed image to: ", outputFileName];
        , Print["Output is not an image: ", images]];
    ];
];

(* Check if the image file exists *)
If[!FileExistsQ[imageFile],
    Print["File not found: ", imageFile];
    Exit[];
];

Print["Processing file: ", imageFile];
(* Load and process the image *)
inputImage = Import[imageFile];

(* Process the image using BiKymoButler with the specified indices *)
resultBI = BiKymoButler[inputImage, .2, .5, "CPU", models["binet"], models["decnet"], 10, 10, True, indicesArg];

(* Save the output images *)
saveImages[resultBI, FileBaseName[imageFile], outputDir];
