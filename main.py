#!/usr/bin/python3

import Lib, OpenCVRunner

def main():
    input = Lib.readImage("Images/cat.jpg")
    runners = [
        OpenCVRunner.OpenCVRunner("Models/opencv/mobilenet_v1_0.25_224_frozen.pb"),
    ]

    for runner in runners:
        output = runner.inference(input)
        print(output)
        Lib.printTopK(output, k=5)



if __name__ == "__main__":
    main()
