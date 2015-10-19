/**
 * For vectors X and Y it computes SUM of (x.i - y.i)^2  for i != dimension 
 */
float getVectDistRestricted(int3 a, int3 b, int d){

    float dist;
	switch(d){
		case 1:
		   	dist = (float)pow((float)a.y-b.y,2) + (float)pow((float)a.z-b.z,2);
			break;
		case 2:
		   	dist = (float)pow((float)a.x-b.x,2) + (float)pow((float)a.z-b.z,2);
			break;
		case 3:
		   	dist = (float)pow((float)a.x-b.x,2) + (float)pow((float)a.y-b.y,2);
			break;
	}

    return dist;
}

/**
 * It computes the L2 norm if 'squared' is false and
 * the L2^2 if 'squared' is true
 */
float getVectDist(int3 a, int3 b, bool squared ){
    float3 c = (float3)(a.x-b.x, a.y-b.y, a.z-b.z);
    float dist;

    if( squared)
        dist =  pow(c.x,2) + pow(c.y,2) + pow(c.z,2);
    else
        dist = sqrt( pow(c.x,2) + pow(c.y,2) + pow(c.z,2));

    return dist;
}

/**
 * Gets the 1D index from a 2D coordinates. (from (row,col) it gives a unique index
 */
int indexFromCoord(int width, int height, int depth, int row, int col, int z){
    return width*height*z + width*row + col;
}

/**
 * It gets the 3 dimension coordinates from 1D index. From index it gives the
 * corresponding row, column and z value
 */
int3 getCoords(int index, int width, int height){

    int3 coords = (int3)(0,0,0);
    coords.z = (int)floor((float) index/(width*height) );// z value 
    coords.x = (int)floor((float) (index - coords.z*width*height)/(width) );// current row
    coords.y = index - coords.z*width*height - coords.x*width;//Current col

    return coords;
}

/**
 * It receives  g[l-2] g[l-1] currIndex currcords
 */
bool removeFV(int ui, int vi, int wi, int3 currentCell, int width, int height, int d){
    int3 u = getCoords(ui,width, height);
    int3 v = getCoords(vi,width, height);
    int3 w = getCoords(wi,width, height);

    float a;
    float b;

    switch(d){
		case 1:
			a = v.x - u.x;
			b = w.x - v.x;
			break;
		case 2:
			a = v.y - u.y;
			b = w.y - v.y;
			break;
		case 3:
			a = v.z - u.z;
			b = w.z - v.z;
			break;
    }

    float c = a + b;

    float test_eqn =	  c*getVectDistRestricted(currentCell, v, d)
                        - b*getVectDistRestricted(currentCell, u, d)
                        - a*getVectDistRestricted(currentCell, w, d)
                        - a*b*c;
	
	//Testing
//	float d1 = getVectDist(currentCell,w,false);
//	float d2 = getVectDist(currentCell,v,false);
//	float d3 = getVectDist(currentCell,u,false);
//    if( (d2 >= d1) && (d2 >= d3) )

    if( test_eqn > 0)
        return true;
    else
        return false;
}

/**
 * This function returns the current 'pseudo row' that we are analyzing. 
 * if we are on dimension 1 then it returns a real row of the image
 * if we are on dimension 2 then it returns a column
 * if we are on dimension 3 then it returns a 'depth row'
 */
int3 getCurrCell(int currRow, int currCol, int currZ, int cell, int d){
    switch(d){
		case 1:
			return (int3)(cell, currCol, currZ);
		case 2:
			return (int3)(currRow, cell, currZ);
		case 3:
			return (int3)(currRow, currCol, cell);
	}
}

/**
* This function merges the two SDF results. buf_phi_half has the distances to
* the closest 0 value and sec_half thas the distances to the closest > 0 value
*/
__kernel void
mergePhisBuf( __global float* buf_phi_half,__global float* buf_phi_sec_half,
			__global float*buf_phi,int width, int height, int depth){
				
	int col = get_global_id(0);
	int row = get_global_id(1);

	for(int z=0; z<depth; z++){
		int currIndex = indexFromCoord(width, height,depth, row, col, z);

		float fhalf = buf_phi_half[currIndex];
		float sechalf = buf_phi_sec_half[currIndex];

		if( sechalf > -1){
//			buf_phi[currIndex] = fhalf - sechalf - .5;
//			buf_phi[currIndex] = fhalf;
			buf_phi[currIndex] = fhalf + (float) min((float) (-sechalf + 1),(float) 0) - .5;
		}else{
			buf_phi[currIndex] = fhalf;
		}
	}
}

/* This function iterates of the second dimension and, in the case of 2D 
   computes the distance to the closes pixel to the oposite binary value */
__kernel void
SDF_voroStep2Buf(__global float* src, __global float* dst, 
		int width, int height, int depth, int d){

	// Maximum possible distance (from one corner to the other, used to normalize
	// the distances from 0 to 255)
	float maxDist = getVectDist( (int3)(0,0,0), (int3)(width,height,depth), false);

    int currCol = (int)get_global_id(0);
    int currRow = (int)get_global_id(1);
    int currZ = (int)get_global_id(2);

    int rwidth = 0;//This will be the line width (dependent of current d)

	// 4096 is close to the limit of private memory
	int g[4096];// TODO THIS ARRAY SHOULD BE THE SIZE OF the dimsension we are working on
	// This array will contain all the possible Feature Pixels for the Voronoi diragram

	//For d==1 internal loop is for rows and for d=2 internal is for columns
    switch(d){
		case 1:
			rwidth = height;
			break;
		case 2:
			rwidth = width;
			break;
		case 3:
			rwidth = depth;
			break;
	}

	int currIndex = 0; //Index used inside the loop, it iterates over rows, cols or depth 

    int l=0;
    float x = 0; //Value that we read from source
	float fij = 0; //Final result shortest distance to voronoi cell on that row

    int3 currentCell = (int3)(0 , 0, 0);//Only used to get the currIndex 

	//This is the internal loop 
	switch(d){//It is repetitive, but faster because we remove one if inside the loop
		case 1:
			for(int cell=0; cell < rwidth; cell++){

				//This is the current cell we are analyzing for d=1 it iterates over a column
				currentCell = (int3)(cell,currCol,currZ);
				// Current cell has first row and then col we need to swap it
				currIndex = indexFromCoord(width, height,depth, currentCell.x, currentCell.y, currentCell.z);

				x = src[currIndex];//Get value of current index

				if( x > -1){
					if( l>1 ){
					   //This is the bottleneck of the program.
					   while( (l>1) && (removeFV( g[l-2] , g[l-1] , x , currentCell ,
											 width , height, d) )) {
							l--;
						}
					}
					g[l] = x;
					l++;
				}
			}
			break;
		case 2:
			for(int cell=0; cell < rwidth; cell++){

				//This is the current cell we are analyzing for d=1 it iterates over a column
				currentCell = (int3)(currRow,cell,currZ);
				// Current cell has first row and then col we need to swap it
				currIndex = indexFromCoord(width, height,depth, currentCell.x, currentCell.y, currentCell.z);

				x = src[currIndex];//Get value of current index

				if( x > -1){
					if( l>1 ){
					   //This is the bottleneck of the program.
					   while( (l>1) && (removeFV( g[l-2] , g[l-1] , x , currentCell ,
											 width , height, d) )) {
							l--;
						}
					}
					g[l] = x;
					l++;
				}
			}
			break;
		case 3:
			for(int cell=0; cell < rwidth; cell++){

				//This is the current cell we are analyzing for d=1 it iterates over a column
				currentCell = (int3)(currRow,currCol,cell);
				// Current cell has first row and then col we need to swap it
				currIndex = indexFromCoord(width, height,depth, currentCell.x, currentCell.y, currentCell.z);

				x = src[currIndex];//Get value of current index

				if( x > -1){
					if( l>1 ){
					   //This is the bottleneck of the program.
					   while( (l>1) && (removeFV( g[l-2] , g[l-1] , x , currentCell ,
											 width , height, d) )) {
							l--;
						}
					}
					g[l] = x;
					l++;
				}
			}
			break;
	}
        
	int totg = l-1;// Save total amount
    l = 0;

    int3 g1 = (int3)(0,0,0);
    int3 g2 = (int3)(0,0,0);
    bool doSquareRoot = false;// Avoids the squared root

    float dist1 = 0;
    float dist2 = 0;
	//totg = 0 means that there is one voronoi cell on this 'row'
    if( totg >= 0){
        for(int cell=0; cell < rwidth; cell++){
            currentCell = getCurrCell(currRow, currCol, currZ, cell, d);
			currIndex = indexFromCoord(width, height, depth, currentCell.x, currentCell.y, currentCell.z);

            if(totg > l){

                //Sets the current CFV and the next FV to see which one is the closest
                if( cell == 0){
                    g1 = getCoords(g[l], width, height);
                    g2 = getCoords(g[l+1], width, height);
                }

                if(  g[l+1] != currIndex ){
                    //(g[l+1] == currIndex) is to try to avoid the comparison
                    dist1 = getVectDist( currentCell , g1, doSquareRoot );//To avoid computing it again
                    dist2 = getVectDist( currentCell , g2, doSquareRoot );
                    while( (l < totg) && (dist1 > dist2)){
                        l = l+1;
                        if( totg > l){
                            g1 = getCoords(g[l] , width, height);
                            g2 = getCoords(g[l+1] , width, height);
                            dist1 = dist2;
                            dist2 = getVectDist( currentCell , g2, doSquareRoot );
                        }
                    }
                }else{//This part of the if is to avoid computations, the normal algorithm is 
				//when if=true
                    l = l+1;
                    if( totg > l){
                        g1 = currentCell;
                        g2 = getCoords(g[l+1] , width, height);
                        dist1 = 0;
                    }
                }
            }// totg > l

			fij = 0;
			if( d < 3){
				fij = g[l];
			}else{  // if d<3

		// Write directly the distance in this case from 0 to max(width,height)
//				fij = max(max(width,height),depth)*getVectDist(currentCell, getCoords(g[l], width,height), false)/maxDist;
				fij = getVectDist(currentCell, getCoords(g[l], width,height), false);
                //fij = dist1;//This is not WORKING check it if you want to use it again
				//fij = g[l];
//                    fij = g[l]; //Used to show the closest index
//                    fij = totg; //Used to show the number of voronoi cell - 1 on this row 
//				fij = ((int2)getCoords(g[l], width)).x;
			}

			dst[currIndex]=fij;
		}
    } else{ // If there is no FV in this row leave all the 'row' with -1's (Unknown FV)
		for(int cell=0; cell < rwidth; cell++){

            currentCell = getCurrCell(currRow, currCol,currZ, cell, d);
			currIndex = indexFromCoord(width, height, depth, currentCell.x, currentCell.y, currZ);
			
			dst[currIndex]=-1;
		}
    }
}


/**
 * Fills a new image with the values of a '1D index' inside all the
 * pixels that has value > 0 if mode == 1 or for all the pixels
 * that has value < 0 if mode == 0
 */
__kernel void
SDF_voroStep1Buf(__global uchar* src, __global float* dst, int maxIndex, int mode)
{
    int currIndex = (int)get_global_id(0);

	if(currIndex <= maxIndex){
		uchar x = src[currIndex];
		if( mode == 1){ // Distance to values >  0
			if(x > 0){
				dst[currIndex] = currIndex;
			}else{
				dst[currIndex] = -1;
			}
		}

		if(mode == 0){ // Distance to values == 0
			if(x == 0){
				dst[currIndex] = currIndex;
			}else{
				dst[currIndex] = -1;
			}
		}
	}
}
