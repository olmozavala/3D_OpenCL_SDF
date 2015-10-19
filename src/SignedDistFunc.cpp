/* 
 * File:   SignedDistFunc.cpp
 * Author: olmozavala
 * 
 * Created on October 10, 2011, 10:08 AM
 */

#define MAXF 1
#define MAXDPHIDT 2
#define EPS .00001

#include "SignedDistFunc.h"
#include <sstream>
#include "debug.h"
#include "MatrixUtils/MatrixUtils.h"
#include <iomanip> //For setting the precision of the printing floats

#define cutImageW 12
#define cutImageH 16
//#define PRINT 1
//#define SAVE 1 //Saves intermediate steps as images

template <class T>
inline std::string to_string(char* prev, const T& t, char* post) {
    std::stringstream ss;
    ss << prev << t << post;
    return ss.str();
}

inline std::string appendStr(char* prev, char* post) {
    std::stringstream ss;
    ss << prev << post;
    return ss.str();
}

SignedDistFunc::SignedDistFunc() {
}

SignedDistFunc::SignedDistFunc(const SignedDistFunc& orig) {
}

SignedDistFunc::~SignedDistFunc() {
}

/**
 * Computes the distance function of a binary image.
 * @param {int} if 1 then we compute the distance to the closest white value, if 0 then to
 * the closest black value
 */
cl::Event SignedDistFunc::voroHalfSDF_3DBuf(int posValues, cl::Buffer& outputBuffer) {

    //Define events used in this function
    cl::Event ev1;// Set index at feature values
    cl::Event ev2;// Obtain partial CFP for cols dim 1
    cl::Event ev3;// Obtain partial CFP for rows dim 2
    cl::Event ev4;// Obtain partial CFP for depth dim 3
    vector<cl::Event> vecEv1;
    vector<cl::Event> vecEv2;
    vector<cl::Event> vecEv3;
    vector<cl::Event> vecEv4;

    //Creates two temporal buffers to store the temporal results.
    cl::Buffer buf_temp = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof (float) *buf_size, 0, &err);

    dout << "*************** Computing HALF voro SDF for I = " << 
			posValues << " ****************" << endl;
    try {
#ifdef DEBUG
        Timer tm_step1(sdf_ts, "Step1");
        queue->finish();
        tm_step1.start(Timer::SELF);
#endif

		int totalValues =  width*height*depth;
		int threadsPerGroup = max_warp_size;
		if(totalValues > threadsPerGroup){
			//Making sure that totalValues is divisible by threadsPerGroup 
			while( totalValues % threadsPerGroup != 0) {
				threadsPerGroup--;
			}
		}else{
			threadsPerGroup=totalValues;
		}

		dout << "Total number of values: " << totalValues << endl;
		dout << "Max warp size: " << max_warp_size << endl;
		dout << "Number of threads per group: " << threadsPerGroup << endl;

        //----------------------  Replacing all values of cell > 0 with its index --------------------
        cl::Kernel kernelSDFVoro1(*program, "SDF_voroStep1Buf");
        kernelSDFVoro1.setArg(0, buf_mask);
        kernelSDFVoro1.setArg(1, buf_temp);
        kernelSDFVoro1.setArg(2, totalValues);
        kernelSDFVoro1.setArg(3, posValues); //Mode 1 is for getting SDF to values > 0
        //Mode 0 is for getting SDF to values < 0

        queue->enqueueNDRangeKernel(
                kernelSDFVoro1,
                cl::NullRange,
                cl::NDRange((size_t) totalValues),
                cl::NDRange((size_t) threadsPerGroup), &vecWriteImage, &ev1);

        vecEv1.push_back(ev1);

//This section is only used for debugging, it reads the temporal results and save them as an image
#ifdef DEBUG
        queue->finish();
        tm_step1.end();
		
        dout << "SDFVoro step 1 finished (all values > 0 with its index)" << endl;
        res = queue->enqueueReadBuffer(buf_temp, CL_TRUE, 0, sizeof (float) *buf_size, 
				(void*) array_buf_out, &vecEv1, 0);

        queue->finish();
        dout << "----------- Printing values of result of step1 (all values > 0 with its index) " << endl;

#ifdef SAVE
		dout << "******** Saving Step 1 for dim:  " << posValues << " *******" << endl;

		char temp[35];
		sprintf(temp, "IntermediateSteps/%d/IndexAnd0s_", posValues);
		string folder = appendStr(save_path, temp);
		ImageManager::write3DImage( (char*) folder.c_str(), array_buf_out, width, height, depth);
#endif
#endif

		// --------------------------- Running by cols ---------------------
        int dimension = 1;

		dout << "Start SDF Step: " << dimension+1 << " for RUN " << posValues << endl;
		ev2  = runStep2(buf_temp, outputBuffer, width, height, depth,
				dimension, vecEv1 , posValues);

        vecEv2.push_back(ev2);

		// --------------------------- Running by rows ---------------------
        dimension = 2;
		dout << "Start SDF Step: " << dimension+1 << " for RUN " << posValues << endl;
		ev3  = runStep2(outputBuffer, buf_temp, width, height, depth,
				dimension, vecEv2 , posValues);

        vecEv3.push_back(ev3);

		// --------------------------- Running by z ---------------------
        dimension = 3;
		dout << "Start SDF Step: " << dimension+1 << " for RUN " << posValues << endl;
		ev4  = runStep2(buf_temp, outputBuffer, width, height, depth,
				dimension, vecEv3 , posValues);

        vecEv4.push_back(ev4);

    } catch (cl::Error ex) {
        cl->printError(ex);
    }
    return ev4;
}

cl::Event SignedDistFunc::runStep2(cl::Buffer& inputBuffer, cl::Buffer& outputBuffer,
		int w, int h, int z, int dim, vector<cl::Event>& vecEvPrev, int posValues){

		cl::Event event;

#ifdef DEBUG
        string name = to_string( (char*)"Step",(dim+1),(char*)"_") ;
        Timer tm_stepn(sdf_ts, name.c_str());
        queue->finish();
        tm_stepn.start(Timer::SELF);
#endif

        //----------------------  Obtains the closest voronoi feature for the first dimension -------------------
        cl::Kernel kernelSDFVoro(*program, "SDF_voroStep2Buf");
        kernelSDFVoro.setArg(0, inputBuffer);
        kernelSDFVoro.setArg(1, outputBuffer);
        kernelSDFVoro.setArg(2, w);
        kernelSDFVoro.setArg(3, h);
        kernelSDFVoro.setArg(4, z);
        kernelSDFVoro.setArg(5, dim);

        int tot_grps_x = 0;
        int tot_grps_y = 0;

		bool print = false;
		switch(dim){
			case 1:
				CLManager::getGroupSize(max_warp_size, width, depth, 
						grp_size_x, grp_size_y, tot_grps_x, tot_grps_y, print);

				queue->enqueueNDRangeKernel(
						kernelSDFVoro,
						cl::NullRange,
						cl::NDRange((size_t) width, (size_t) 1, (size_t) depth) ,
						cl::NDRange((size_t) grp_size_x, (size_t) 1, (size_t) grp_size_y), &vecEvPrev, &event);
				break;
			case 2:
				CLManager::getGroupSize(max_warp_size, height, depth, 
						grp_size_x, grp_size_y, tot_grps_x, tot_grps_y, print);

				queue->enqueueNDRangeKernel(
						kernelSDFVoro,
						cl::NullRange,
						cl::NDRange((size_t) 1, (size_t) height, (size_t) depth) ,
						cl::NDRange((size_t) 1, (size_t) grp_size_x,(size_t)  grp_size_y), &vecEvPrev, &event);
				break;
			case 3:
				CLManager::getGroupSize(max_warp_size, width, height, 
						grp_size_x, grp_size_y, tot_grps_x, tot_grps_y, print);

				queue->enqueueNDRangeKernel(
						kernelSDFVoro,
						cl::NullRange,
						cl::NDRange((size_t) width, (size_t) height, (size_t) 1) ,
						cl::NDRange((size_t) grp_size_x, (size_t) grp_size_y, (size_t) 1), &vecEvPrev, &event);
				break;

		}

#ifdef DEBUG

		vector<cl::Event> vecEvPrint;
        vecEvPrint.push_back(event);

        queue->finish();
        tm_stepn.end();

        dout << "******* Step " << dim+1 << " finished CFV for RUN:  " << posValues << endl;
        res = queue->enqueueReadBuffer(outputBuffer, CL_TRUE, 0, sizeof (float) *buf_size, (void*) array_buf_out, &vecEvPrint, 0);
        queue->finish();
#ifdef PRINT
        dout << "----------- Printing values of result of step 2 For dimension: " << dim << endl;
        MatrixUtils<float>::print3DImage(width, height, depth,  array_buf_out);
#endif
#endif

		return event;
}

/**
 * Computes the SDF function using the Voronoi (WHICH) method.
 * It computes the SDF for the binary image obtained from the mas and
 * then it does it again by changing the colors of the binary image
 */
cl::Event SignedDistFunc::SDF3DVoroBuf() {

    try {
        // Computes the distance from all pixels to the closest pixel == 0
        cl::Event lastNegEvent = voroHalfSDF_3DBuf(0, buf_sdf_half2);

        // Computes the distance from all pixels to the closest pixel > 0
        cl::Event lastPosEvent = voroHalfSDF_3DBuf(1, buf_sdf_half);


#ifdef DEBUG
        Timer tm_merge(sdf_ts, "Merging");
        queue->finish();
        tm_merge.start(Timer::SELF);
#endif

		int tot_grps_x, tot_grps_y; 
        dout << "Merging PHIs........." << endl;

		bool print = false;//Do not print the results
		CLManager::getGroupSize(max_warp_size, width, height, 
				grp_size_x, grp_size_y, tot_grps_x, tot_grps_y, print);

        dout << width << " x " << height << endl;
        dout << "Work group size: " << grp_size_x << " x " << grp_size_y << endl;

        // Merges the two buffers sdf_half and sdf_half2 into the final SDF with 
        // negative distances for pixels > 0
        kernelMergePhis = cl::Kernel(*program, "mergePhisBuf");
        kernelMergePhis.setArg(0, buf_sdf_half);
        kernelMergePhis.setArg(1, buf_sdf_half2);
        kernelMergePhis.setArg(2, buf_sdf);
        kernelMergePhis.setArg(3, width);
        kernelMergePhis.setArg(4, height);
        kernelMergePhis.setArg(5, depth);

        vector<cl::Event> prevEvents;
        prevEvents.push_back(lastNegEvent);
        prevEvents.push_back(lastPosEvent);

        queue->enqueueNDRangeKernel(
                kernelMergePhis,
                cl::NullRange,
                cl::NDRange((size_t) width, (size_t) height),
                cl::NDRange((size_t) grp_size_x, (size_t) grp_size_y), &prevEvents, &evEndSDF);

        vector<cl::Event> vecEvMerg;
        vecEvMerg.push_back(evEndSDF);

#ifdef DEBUG
        tm_merge.end();

        queue->finish();

		int buf_size = width*height*depth;
        res = queue->enqueueReadBuffer(buf_sdf, CL_TRUE, 0, sizeof (float) *buf_size, (void*) array_buf_out, &vecEvMerg, 0);


#ifdef PRINT
        queue->finish();
        dout << "----------- Printing SDF yeah!! ------------" << endl;
        MatrixUtils<float>::print3DImage(width, height, depth,  array_buf_out);
#endif
#endif

    } catch (cl::Error ex) {
        cl->printError(ex);
    }
    return evEndSDF;
}

/**
 * This is the main function in charge of computing the 3D Signed Distance function of a 
 * 3D binary mask
 * @param clOrig A CLManager object with an initialized CL context, queue and device selected
 * @param buf_mask Initial binary mask where the SDF will be computed 
 * @param buf_sdf  Buffer where the SDF will be set as an output 
 * @param max_warp_size 
 * @param width
 * @param height
 * @param depth
 * @param evWrtImg It has the event that is required to wait for writing the buffers
 * @param save_path
 * @return 
 */
cl::Event SignedDistFunc::run3DSDFBuf(CLManager* clOrig,  cl::Buffer& buf_mask, cl::Buffer& buf_sdf, 
		int max_warp_size, int width, int height, int depth, cl::Event evWrtImg, char* save_path) {
	
	dout << "--------- Running SDF in 3D with dimmensions: " << width 
			<< ", " << height << ", " << depth << " ----------" << endl;
#ifdef DEBUG
    Timer tm_prev(sdf_ts, "Prev");
    tm_prev.start(Timer::SELF);
#endif
	
    this->save_path = save_path;
    this->buf_mask = buf_mask;
    this->buf_sdf = buf_sdf;
    cl = clOrig;
	
    buf_size = width*height*depth;
	
    try {
		
        context = cl->getContext();
        queue = cl->getQueue();
        program = cl->getProgram();
		
        array_buf_out = new float[buf_size];
		
        this->max_warp_size = max_warp_size;
		
        this->width = width;
        this->height = height;
        this->depth = depth;
		
        origin = CLManager::getSizeT(0, 0, 0);
        region = CLManager::getSizeT(width, height, depth);
		
		//These buffers will have the SDF to closest 0 value and to clostest >0 value
        buf_sdf_half = cl::Buffer(*context, CL_MEM_READ_WRITE, buf_size * sizeof (float), NULL, &err);
        buf_sdf_half2 = cl::Buffer(*context, CL_MEM_READ_WRITE, buf_size * sizeof (float), NULL, &err);
		
#ifdef DEBUG
        queue->finish();
        tm_prev.end();
#endif
        vecWriteImage.push_back(evWrtImg);

		evEndSDF = SDF3DVoroBuf();
#ifdef DEBUG
        sdf_ts.dumpTimings();
#endif
    } catch (cl::Error ex) {
        cl->printError(ex);
    }
	
    delete[] array_buf_out;
	
    return evEndSDF; // Delete is just for testing
}
