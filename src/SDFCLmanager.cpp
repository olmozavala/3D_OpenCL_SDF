/* 
 * File:   SDFCLmanager.cpp
 * Author: olmozavala
 * 
 * Created on November 24, 2011, 9:06 PM
 */

#include "CLManager/CLManager.h"
#include "SDFCLmanager.h"
#include "SignedDistFunc.h"
#include <sstream>
#include "CL/cl.hpp"
#include "debug.h"
#include <iomanip>
#include <nifti1_io.h>
#include "MatrixUtils/MatrixUtils.h"

#define MAXF 1
#define MAXDPHIDT 2
#define EPS .00001

// These two variables define how many pixels display as a matrix
// when debugging and printing values
#define cutImageW 12
#define cutImageH 16

SDFCLmanager::SDFCLmanager() {
}

SDFCLmanager::SDFCLmanager(const SDFCLmanager& orig) {
}

SDFCLmanager::~SDFCLmanager() {
}

/**
 * This function creates a tag with dimensions widthxheightxdepth and with
 * a 'cube' as an initial ROI
 */
void SDFCLmanager::create3DMask(int width, int height, int depth,
        int colStart, int colEnd, int rowStart, int rowEnd, int depthStart, int depthEnd) {

    arr_buf_mask = new unsigned char[width * height * depth];

    //Update local width and height of the image
    width = width;
    height = height;
    depth  = depth;

    int size = width * height * depth;
    int indx = 0;

    dout << "--------------------- Creating mask (" << width << "," << height << "," << depth << ") ----------" << endl;

    dout << "col min: " << colStart << " col max: " << colEnd << endl;
    dout << "row min: " << rowStart << " row max: " << rowEnd << endl;
    dout << "depth min: " << depthStart << " depth max: " << depthEnd << endl;

    //Initialize to 0
    for (int i = 0; i < size; i++) {
        arr_buf_mask[i] = 0; // Red value
    }

    int count = 0;
    //Set the internal mask to 1
    for (int z = depthStart; z < depthEnd; z++) {
        for (int row = rowStart; row < rowEnd; row++) {
            //indx = ImageManager::indxFromCoord3D(width, height, row, rowStart, z);
            indx = width * height * z + width * row + colStart;
            for (int col = colStart; col < colEnd; col++) {
                arr_buf_mask[indx] = 1; //R
                indx = indx + 1;
                count++;
            }
        }
    }

    dout << "Total one's on mask: " << count << endl;
}

/**
 * This is the main function used to compute a 3D SDF using gif images as initial masks
 * @param inputFile	Name of the input file
 * @param outputFolder Folder where to put the SDF result as a set of images
 * @return 
 */
int SDFCLmanager::run3dBuf(char* inputFile, char* outputFolder) {

//	ImageManager::printVariablesSizes();
    //Create timers
    Timings ts;
    Timer tm_context(ts, "ContextAndLoad");
    Timer tm_sdf(ts, "SDF_kernels");
    Timer tm_all(ts, "All_time");

    int width, height, depth;

    float* array_buf_out;

    tm_all.start(Timer::SELF); // Start time for ALLL

    if(is_nifti_file(inputFile)){
        cout << "------------------------" << endl << endl << inputFile<< " is a Niftifile" << endl;
        bool readData = true;
        nifti_image* image = nifti_image_read(inputFile, readData);
        
        width = image->dim[1];
        height = image->dim[2];
        depth = image->dim[3];
        
        int size = image->nvox;
        
        array_buf_out = new float[size];
        array_buf_out = (float*)image->data;

    }else{
        cout << inputFile << " is NOT a Niftifile" << endl;
        //Loads the input image and the values for width and height and depth
        arr_buf_mask = ImageManager::load3dImageGif(inputFile, width, height, depth);
        #ifdef PRINT 
            MatrixUtils<unsigned char>::print3DImage(width, height, depth,  arr_buf_mask);
        #endif
    }

	dout<< "Image has been loaded correctly... " << endl;
    #ifdef PRINT 
        //MatrixUtils<unsigned char>::print3DImage(width, height, depth,  arr_buf_mask);
        //READ!!!!! I don't know why but it prints double the size, it seams the print function
        //do not uses  the proper size of the array types
        MatrixUtils<float>::print3DImage(width, height, depth, 16, 32, 10, array_buf_out);
    #endif

    tm_context.start(Timer::SELF);//Start time for context loading
    try {

        // Create the program from source
        cl.initContext(false); //false = not using OpenGL
		cl.addSource((char*) "src/resources/SDFVoroBuf3D.cl");

        // Initialize OpenCL queue, context and program
        cl.initQueue();
        context = cl.getContext();
        queue = cl.getQueue();
        program = cl.getProgram();

        int max_warp_size = 0; 
        max_warp_size = cl.getMaxWorkGroupSize(0);
        //If we leave the original value 1024 it crashes not sure why
        max_warp_size = 512;

		#ifdef DEBUG
			cl.getDeviceInfo(0);
		#endif

		//START******************** FORCING TO HAVE THE SAME MASK AS ACWE
        /*
		int* mask = new int[6];
		mask[0] = 7;//colStart
		mask[1] = 12;//colEnd
		mask[2] = 7;//rowStart 
		mask[3] = 12;//rowEnd
		mask[4] = 10;//depthStart 
		mask[5] = 15;//depthStart 
		
		cout << "Mask cube limits: " << 
                mask[0] << ',' << mask[1]<< ',' << mask[2]<< ',' 
                << mask[3]<< ',' << mask[4]<< ',' << mask[5] << ',' << endl;
		
		width = 20;
		height = 20;
		depth = 20;

		this->create3DMask(width, height, depth, mask[0], mask[1],
				mask[2], mask[3], mask[4], mask[5]);

        width = 20;
        height = 20;
        depth = 20;
                */
		//END******************** FORCING TO HAVE THE SAME MASK AS ACWE
        int* mask = new int[6];
        int maskWidthSize= floor(width/8);
        int maskHeightSize= floor(height/8);
        int maskDepthSize= floor(depth/8);

        mask[0] = floor(width/2-maskWidthSize);//colStart
        mask[1] = floor(width/2+maskWidthSize);//colEnd
        mask[2] = floor(height/2-maskHeightSize);//rowStart 
        mask[3] = floor(height/2+maskHeightSize);//rowEnd
        mask[4] = floor(depth/2-maskDepthSize);//depthStart 
        mask[5] = floor(depth/2+maskDepthSize);//depthStart 

        cout << "Mask cube limits: " << 
            mask[0] << ',' << mask[1]<< ',' << mask[2]<< ',' 
            << mask[3]<< ',' << mask[4]<< ',' << mask[5] << ',' << endl;

        tm_context.end(); //Finish timer for context creation
        queue->finish();// Wait for the image to be written

        tm_sdf.start(Timer::SELF);//Starts time for kernels
        this->create3DMask(width, height, depth, mask[0], mask[1],
                mask[2], mask[3], mask[4], mask[5]);

        dout << endl << "Image size: " << width << " x " << height << " x " << depth << endl;

        int buf_size = width*height*depth;
        array_buf_out = new float[buf_size];

        //Creates two buffers, one for the input binary image and one for the SDF
        cl::Buffer buf_mask = cl::Buffer(*context, CL_MEM_READ_WRITE, buf_size * sizeof (unsigned char), NULL, &err);
        cl::Buffer buf_sdf = cl::Buffer(*context, CL_MEM_WRITE_ONLY, buf_size * sizeof (float), NULL, &err);

        cl::Event evWrtImg;
        dout << "Writing image..." << endl;
        //Writes the input binary image into the buffer

        err = queue->enqueueWriteBuffer(buf_mask, CL_TRUE, 0, sizeof (unsigned char) *buf_size, (void*) arr_buf_mask, 0, &evWrtImg);

        cl::Event evSDF;

        dout << "Running .... " << endl << endl;
        SignedDistFunc sdfObj;

        evSDF = sdfObj.run3DSDFBuf(&cl, buf_mask, buf_sdf, max_warp_size, width, height, depth, evWrtImg, outputFolder);

        queue->finish();
        tm_sdf.end();//Ends time for kernels

        vector<cl::Event> vecFinishSDF;
        vecFinishSDF.push_back(evSDF);

        dout << "Writing Signed Distance Function" << endl;
        res = queue->enqueueReadBuffer(buf_sdf, CL_TRUE, 0, sizeof (float) *buf_size, (void*) array_buf_out, &vecFinishSDF, 0);

        dout << "Writing images at: " << outputFolder << endl;

        //START ******************** FORCING TO HAVE THE SAME MASK AS ACWE
        /*
        cout << std::setprecision(3) << endl;
        this->printBuffer(buf_sdf, 400, width*height*9, width, height, vecFinishSDF);
        this->printBuffer(buf_sdf, 400, width*height*10, width, height, vecFinishSDF);
        this->printBuffer(buf_sdf, 400, width*height*11, width, height, vecFinishSDF);
        */
        //END******************** FORCING TO HAVE THE SAME MASK AS ACWE

#ifdef PRINT 
        ImageManager::write3DImageSDF((char*) outputFolder, array_buf_out, width, height, depth);
#endif

        dout << "SUCESS!!!" << endl;
        tm_all.end();
        ts.dumpTimings();
    } catch (cl::Error ex) {
        cl.printError(ex);

        return EXIT_FAILURE;
    }

    delete[] arr_buf_mask;
    delete[] array_buf_out;

    return EXIT_SUCCESS;
}

void SDFCLmanager::printBuffer(cl::Buffer& buf, int size, int offset, int width, int height, vector<cl::Event> vecPrev){

    float* result = new float[size];
    dout << "Reading: " << size<< " elements" << endl;
    // buffer, block, offset, size, ptr, ev_wait, new_ev
    res = queue->enqueueReadBuffer
        (buf, CL_TRUE, (size_t) (sizeof(float)*offset), (size_t) (sizeof(float)*size), 
         (void*) result, &vecPrev, 0);

    queue->finish();

    int count = 0;
    while(count < size){
        cout << "------ Slice ------" << (offset/(width*height) + 1) << endl << endl;
        for(int row= 0; row<height; row++){
            for(int col= 0; col<width; col++){
                if( count < size){
                    cout << result[count] << "\t";
                    count++;
                }else{
                    break;
                }
            }//col
            cout << endl;
            if( count >= size){
                break;
            }
        }//row
    }
}
