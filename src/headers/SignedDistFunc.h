/* 
 * File:   SignedDistFunc.h
 * Author: olmozavala
 *
 * Created on October 10, 2011, 10:08 AM
 */

#ifndef SIGNEDISTFUNC_H 
#define	SIGNEDISTFUNC_H 

#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>

#include "FreeImage.h"
#include "CLManager/CLManager.h"
#include "FileManager/FileManager.h"
#include "ImageManager/ImageManager.h"
#include "CLManager/ErrorCodes.h"
#include "Timers/timing.h"

#define SDFOZ 0
#define SDFVORO 1

class SignedDistFunc {
public:
    SignedDistFunc();
    SignedDistFunc(const SignedDistFunc& orig);
    virtual ~SignedDistFunc();

	cl::Event run3DSDFBuf(CLManager* cl, cl::Buffer& buf_mask, cl::Buffer& buf_sdf, 
			int max_warp_size, int width, int height, int depth, cl::Event evWrtImg,char* save_path);
private:
	cl::Event SDF3DVoroBuf();

    cl::Event voroHalfSDF_3DBuf(int posValues, cl::Buffer& outputBuffer);//Using buffers

	cl::Event runStep2(cl::Buffer& inputBuffer, cl::Buffer& outputBuffer,
		int w, int h, int z, int dim, vector<cl::Event>& vecEvPrev, int posValues);

	CLManager* cl;
		
    cl::Context* context;
    cl::CommandQueue* queue;
    cl::Program* program;

    //--- Kernels
	cl::Kernel kernelMergePhis;

    //--- Events
	vector<cl::Event> vecWriteImage;// Events that controls the writing of images
    cl::Event evEndSDF;

    float* array_buf_out;

	//--------------------- USING BUFFERS -------------------------
	//Buffers used for the SDF algorithm
    cl::Buffer buf_mask;
    cl::Buffer buf_sdf;
    cl::Buffer buf_sdf_half;
    cl::Buffer buf_sdf_half2;

    int width, height, depth;
	int buf_size;
    int grp_size_x;
    int grp_size_y;

    int max_warp_size;
	cl::size_t<3> origin;
	cl::size_t<3> region;

    cl_int err;
    cl_int res;

    Timings sdf_ts;

	char* save_path;
};

#endif	/* ACTIVECONTOURS_H */

