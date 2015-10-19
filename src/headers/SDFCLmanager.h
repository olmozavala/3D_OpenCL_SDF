/* 
 * File:   SDFCLmanager.h
 * Author: olmozavala
 *
 * Created on November 24, 2011, 9:06 PM
 */

#ifndef SDFCLMANAGER_H
#define	SDFCLMANAGER_H

#include <CL/cl.hpp>
#include <sstream>

class SDFCLmanager {
public:
    SDFCLmanager();
    SDFCLmanager(const SDFCLmanager& orig);
    virtual ~SDFCLmanager();
    int run3dBuf( char* inputFile, char* outputFile);
	void create3DMask(int width, int height, int depth,
        int colStart, int colEnd, int rowStart, int rowEnd, int depthStart, int depthEnd);
	void printBuffer(cl::Buffer& buf, int size, int offset, int width, int height, vector<cl::Event> vecPrev);
private:
		
    cl_int err;
    cl_int res;

    //This is a helper class for OpenCL
    CLManager cl;

    cl::Context* context;
    cl::CommandQueue* queue;
    cl::Program* program;

    unsigned char* arr_buf_mask;

};

#endif	/* SDFCLMANAGER_H */

