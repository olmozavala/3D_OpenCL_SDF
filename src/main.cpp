
#define __CL_ENABLE_EXCEPTIONS

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include "FreeImage.h"
#include "SignedDistFunc.h"
#include "SDFCLmanager.h"

using namespace std;

inline std::string appendStr(char* prev, char* post) {
    std::stringstream ss;
    ss << prev << post;
    return ss.str();
}

/**
 * This function simple selects examples from the image folder
 */
void selectExample(int example, char** inputImage, char** outputImage){

    switch (example) {
        case 7:
            *inputImage = (char*) "images/3Dimages/BasicSquare.gif";
            *outputImage = (char*) "images/3Dimages/Results/";
			break;
		case 8:
            *inputImage = (char*) "images/3Dimages/BigSquare.gif";
            *outputImage = (char*) "images/3Dimages/Results/";
			break;
		case 9:
            *inputImage = (char*) "images/3Dimages/BigCircle.gif";
            *outputImage = (char*) "images/3Dimages/Results/";
			break;
		case 10:
            *inputImage = (char*) "images/3Dimages/1kExample.gif";
            *outputImage = (char*) "images/3Dimages/Results/";
			break;
        case 11:
            *inputImage = (char*) "/home/olmozavala/Dropbox/TestImages/nifti/Basics/Box80.nii";
            *outputImage = (char*) "images/3Dimages/Results/";
			break;		
    }
	
}

/**
 * This is the main function, it simply runs 
 * the SDF with an specific example or the one received as parameter
 */
int main(int argc, char** args){

    char* inputImage;
    char* outputImage;
	
    bool perf_tests = false;

    SDFCLmanager ac = SDFCLmanager();

    if(perf_tests){

        int example = 11; //Example to use, by default is 1
        if(argc < 2){
            cout<<"Please select an example from 1 to 6. Using "<< example <<" as default" << endl;
        }else{
            example= atoi(args[1]);
        }

        //Selects the example we want to use
        selectExample(example, &inputImage, &outputImage);
        inputImage = (char*) "/home/olmozavala/Dropbox/TestImages/nifti/Basics/Gradient240.nii";
        //Without performance tests, (normal runs)
        return ac.run3dBuf((char*) inputImage, (char*) outputImage); //With buffers
    }else{
        //Performance tests
        int size = 15;

        string files[15] = {"Gradient32.nii" ,"Gradient64.nii" ,"Gradient96.nii" ,"Gradient128.nii" ,"Gradient160.nii" ,"Gradient192.nii" ,"Gradient224.nii" ,"Gradient256.nii" ,"Gradient288.nii" ,"Gradient320.nii" ,"Gradient352.nii" ,"Gradient384.nii" ,"Gradient416.nii" ,"Gradient448.nii" ,"Gradient480.nii"};

        string input_path = "/home/olmozavala/Dropbox/TestImages/nifti/Basics/";
        string output_path = "/home/olmozavala/Dropbox/OzOpenCL/SignedDistanceFunction/images/TestSizes/";


        for (int i = 0; i < size; i++) {
            string inputImage = appendStr((char*) input_path.c_str(), (char*) files[i].c_str());
            string outputImage = appendStr((char*) output_path.c_str(), (char*) files[i].c_str());
            ac.run3dBuf((char*) inputImage.c_str(), (char*) outputImage.c_str()); //With buffers
        }
    }

}
