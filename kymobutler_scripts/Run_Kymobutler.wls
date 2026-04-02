#!/usr/bin/env wolframscript
(* ::Package:: *)

(* Determine the directory of the script or the notebook *)
dir = DirectoryName@$InputFileName;
If[dir == "", dir = NotebookDirectory[]];

(* Load necessary packages *)
Get[FileNameJoin@{dir, "KymoButler_greyscale.wl"}];
Get[FileNameJoin@{dir, "KymoButler_greyscale.wl"}];

Get[FileNameJoin@{dir, "KymoButlerPProc.wl"}];

(* Set random seed for deterministic results *)
SeedRandom[1234];

(* Load the default neural networks models *)
models = Quiet[loadDefaultNets[dir <> ".."]];

(* Get command line arguments *)
args = Rest[$ScriptCommandLine];

(* Parse the output directory and list of indices *)
If[Length[args] >= 1,
    imageFile = First[args];
    outputDir = FileNameJoin@{DirectoryName[imageFile], FileBaseName[imageFile] <> "_output"};
    indicesArg = None;  (* Adjust if necessary *)
,
    Print["Usage: wolframscript -script Run_KymoButler_updated_greyscale.wls <inputFile>"];
    Exit[1];
];
Print["Output Dir ", outputDir];

(* Create output directory if it does not exist *)
If[!DirectoryQ[outputDir], CreateDirectory[outputDir]];

(* Process the image file *)
If[Not@FileExistsQ[imageFile],
    Print["File not found: ", imageFile];
    Exit[1];
];

Print["Processing file: ", imageFile];

(* Load and process the image *)
inputImage = Import[imageFile];

(* Define the saveImages function before using it *)
saveImages[results_, baseName_, dir_] := Module[{outputFileName, image},
    If[Length[results] >= 4,
        (
            (* Save the overlay image with tracks overlaid on the original colored image *)
            image = results[[4]];  (* The overlay image *)
            If[ImageQ[image],
                (
                    outputFileName = FileNameJoin@{dir, baseName <> "_overlay.png"};
                    Export[outputFileName, image];
                    Print["Saved overlay image to: ", outputFileName];
                ),
                (* Else part for ImageQ *)
                Print["Overlay image is not a valid image."]
            ];
        ),
        (* Else part for Length[results] *)
        Print["Results do not contain enough elements to save images."]
    ];
];

(* Run the KymoButler analysis *)
resultBI = BiKymoButler[inputImage, .2, .5, "CPU", models["binet"], models["decnet"], 10, 10, True];
Print["Bidirectional Result for ", imageFile, ": ", resultBI];

(* Save the overlay image *)
saveImages[resultBI, FileBaseName[imageFile], outputDir];

