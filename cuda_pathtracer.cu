/*
*  CUDA based triangle mesh path tracer using BVH acceleration by Sam lapere, 2016
*  BVH implementation based on real-time CUDA ray tracer by Thanassis Tsiodras, 
*  http://users.softlab.ntua.gr/~ttsiod/cudarenderer-BVH.html 
*  Interactive camera with depth of field based on CUDA path tracer code 
*  by Peter Kutz and Yining Karl Li, https://github.com/peterkutz/GPUPathTracer
*
*  This program is free software; you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation; either version 2 of the License, or
*  (at your option) any later version.
*
*  This program is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with this program; if not, write to the Free Software
*  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/
 
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cuda.h>
#include <math_functions.h>
#include <vector_types.h>
#include <vector_functions.h>
#include "device_launch_parameters.h"
#include "cutil_math.h"
#include "C:\Program Files\NVIDIA Corporation\Installer2\CUDASamples_7.5.{075424A8-24ED-4D5A-B1EB-F0B5B2EDDCFB}\common\inc\GL\glew.h"
#include "C:\Program Files\NVIDIA Corporation\Installer2\CUDASamples_7.5.{075424A8-24ED-4D5A-B1EB-F0B5B2EDDCFB}\common\inc\GL\freeglut.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>

#include "cuda_pathtracer.h"

#define M_PI 3.1415926535897932384626422832795028841971f
#define TWO_PI 6.2831853071795864769252867665590057683943f
#define NUDGE_FACTOR     1e-3f  // epsilon
#define samps  1 // samples
#define BVH_STACK_SIZE 32
#define SCREEN_DIST (height*2)

int texturewidth = 0;
int textureheight = 0;
int total_number_of_triangles;

__device__ int depth = 0;


// Textures for vertices, triangles and BVH data
// (see CudaRender() below, as well as main() to see the data setup process)
texture<uint1, 1, cudaReadModeElementType> g_triIdxListTexture;
texture<float2, 1, cudaReadModeElementType> g_pCFBVHlimitsTexture;
texture<uint4, 1, cudaReadModeElementType> g_pCFBVHindexesOrTrilistsTexture;
texture<float4, 1, cudaReadModeElementType> g_trianglesTexture;

Vertex* cudaVertices;
float* cudaTriangleIntersectionData;
int* cudaTriIdxList = NULL;
float* cudaBVHlimits = NULL;
int* cudaBVHindexesOrTrilists = NULL;
Triangle* cudaTriangles = NULL;
Camera* cudaRendercam = NULL;


struct Ray {
	float3 orig;	// ray origin
	float3 dir;		// ray direction	
	__device__ Ray(float3 o_, float3 d_) : orig(o_), dir(d_) {}
	__device__ Ray(Vector3Df o_, Vector3Df d_) : 
		orig(make_float3(o_.x, o_.y, o_.z)), 
		dir(make_float3(d_.x, d_.y, d_.z)) {}
};

enum Refl_t { DIFF, METAL, SPEC, REFR, COAT };  // material types

struct Sphere {

	float rad, emit;				// radius 
	float3 pos;	// position, emission, color 
	//Refl_t refl;			// reflection type (DIFFuse, SPECular, REFRactive)

	__device__ float intersect(const Ray &r) const { // returns distance, 0 if nohit 

		// Ray/sphere intersection
		// Quadratic formula required to solve ax^2 + bx + c = 0 
		// Solution x = (-b +- sqrt(b*b - 4ac)) / 2a
		// Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0 

		float3 op = pos - r.orig;  // 
		float t, epsilon = 0.01f;
		float b = dot(op, r.dir);
		float disc = b*b - dot(op, op) + rad*rad; // discriminant
		if (disc<0) return 0; else disc = sqrtf(disc);
		return (t = b - disc)>epsilon ? t : ((t = b + disc)>epsilon ? t : 0);
	}

};

__device__ Sphere spheres[] = {

	// sun
	//{ 1e5f, .3f, { 1e5f + 700.f, 0.f, 0.f }},  // 37, 34, 30  X: links rechts Y: op neer
	// sky
	{ 20.f, .05f, { 0.f, 0.f, 0.f }},
	// ground
	//{ 100000.f, 0.f, { 0.0f, -100001.2, 0.f }},


	//{ .5f, 0.f, {0.f, 0.f, 0.f } },

	// mountains
	//{ 4e4, { 50.0f, -4e4 - 30, -3000 }, { 0, 0, 0 }, { 0.2f, 0.2f, 0.2f }, DIFF },
	// white Mirr
	// { 1.1, { 0, 0, -2 }, { 0, 0.0, 0 }, { 0.9f, .9f, 0.9f }, SPEC }
	// Glass
	//{ 0.3, { 0.0f, -0.4, 4 }, { .0, 0., .0 }, { 0.9f, 0.9f, 0.9f }, DIFF },
	// Glass2
	//{ 22, { 87.0f, 22, 24 }, { 0, 0, 0 }, { 0.9f, 0.9f, 0.9f }, SPEC },
};


// Create OpenGL BGR value for assignment in OpenGL VBO buffer
__device__ int getColor(Vector3Df& p)  // converts Vector3Df colour to int
{
	return (((unsigned)p.z) << 16) | (((unsigned)p.y) << 8) | (((unsigned)p.x));
}

// Helper function, that checks whether a ray intersects a bounding box (BVH node)
__device__ bool RayIntersectsBox(const Vector3Df& originInWorldSpace, const Vector3Df& rayInWorldSpace, int boxIdx)
{
	// set Tnear = - infinity, Tfar = infinity
	//
	// For each pair of planes P associated with X, Y, and Z do:
	//     (example using X planes)
	//     if direction Xd = 0 then the ray is parallel to the X planes, so
	//         if origin Xo is not between the slabs ( Xo < Xl or Xo > Xh) then
	//             return false
	//     else, if the ray is not parallel to the plane then
	//     begin
	//         compute the intersection distance of the planes
	//         T1 = (Xl - Xo) / Xd
	//         T2 = (Xh - Xo) / Xd
	//         If T1 > T2 swap (T1, T2) /* since T1 intersection with near plane */
	//         If T1 > Tnear set Tnear =T1 /* want largest Tnear */
	//         If T2 < Tfar set Tfar="T2" /* want smallest Tfar */
	//         If Tnear > Tfar box is missed so
	//             return false
	//         If Tfar < 0 box is behind ray
	//             return false
	//     end
	// end of for loop

	float Tnear, Tfar;
	Tnear = -FLT_MAX;
	Tfar = FLT_MAX;

	float2 limits;

// box intersection routine
#define CHECK_NEAR_AND_FAR_INTERSECTION(c)							    \
    if (rayInWorldSpace.##c == 0.f) {						    \
	if (originInWorldSpace.##c < limits.x) return false;					    \
	if (originInWorldSpace.##c > limits.y) return false;					    \
	} else {											    \
	float T1 = (limits.x - originInWorldSpace.##c)/rayInWorldSpace.##c;			    \
	float T2 = (limits.y - originInWorldSpace.##c)/rayInWorldSpace.##c;			    \
	if (T1>T2) { float tmp=T1; T1=T2; T2=tmp; }						    \
	if (T1 > Tnear) Tnear = T1;								    \
	if (T2 < Tfar)  Tfar = T2;								    \
	if (Tnear > Tfar)	return false;									    \
	if (Tfar < 0.f)	return false;									    \
	}

	limits = tex1Dfetch(g_pCFBVHlimitsTexture, 3 * boxIdx); // box.bottom._x/top._x placed in limits.x/limits.y
	//limits = make_float2(cudaBVHlimits[6 * boxIdx + 0], cudaBVHlimits[6 * boxIdx + 1]);
	CHECK_NEAR_AND_FAR_INTERSECTION(x)
	limits = tex1Dfetch(g_pCFBVHlimitsTexture, 3 * boxIdx + 1); // box.bottom._y/top._y placed in limits.x/limits.y
	//limits = make_float2(cudaBVHlimits[6 * boxIdx + 2], cudaBVHlimits[6 * boxIdx + 3]);
	CHECK_NEAR_AND_FAR_INTERSECTION(y)
	limits = tex1Dfetch(g_pCFBVHlimitsTexture, 3 * boxIdx + 2); // box.bottom._z/top._z placed in limits.x/limits.y
	//limits = make_float2(cudaBVHlimits[6 * boxIdx + 4], cudaBVHlimits[6 * boxIdx + 5]);
	CHECK_NEAR_AND_FAR_INTERSECTION(z)

	// If Box survived all above tests, return true with intersection point Tnear and exit point Tfar.
	return true;
}


//////////////////////////////////////////
//	BVH intersection routine	//
//	using CUDA texture memory	//
//////////////////////////////////////////

// there are 3 forms of the BVH: a "pure" BVH, a cache-friendly BVH (taking up less memory space than the pure BVH)
// and a "textured" BVH which stores its data in CUDA texture memory (which is cached). The last one is gives the 
// best performance and is used here.

__device__ bool BVH_IntersectTriangles(
	int* cudaBVHindexesOrTrilists, const Vector3Df& origin, const Vector3Df& ray, unsigned avoidSelf,
	int& pBestTriIdx, Vector3Df& pointHitInWorldSpace, float& hitdist,
	float* cudaBVHlimits, float* cudaTriangleIntersectionData, int* cudaTriIdxList, Vector3Df& boxnormal)
{
	// in the loop below, maintain the closest triangle and the point where we hit it:
	pBestTriIdx = -1;
	float bestTriDist;

	// start from infinity
	bestTriDist = FLT_MAX;

	// create a stack for each ray
	// the stack is just a fixed size array of indices to BVH nodes
	int stack[BVH_STACK_SIZE];
	
	int stackIdx = 0;
	stack[stackIdx++] = 0; 
	Vector3Df hitpoint;

	// while the stack is not empty
	while (stackIdx) {
		
		// pop a BVH node (or AABB, Axis Aligned Bounding Box) from the stack
		int boxIdx = stack[stackIdx - 1];
		//uint* pCurrent = &cudaBVHindexesOrTrilists[boxIdx]; 
		
		// decrement the stackindex
		stackIdx--;

		// fetch the data (indices to childnodes or index in triangle list + trianglecount) associated with this node
		uint4 data = tex1Dfetch(g_pCFBVHindexesOrTrilistsTexture, boxIdx);

		// texture memory BVH form...
		// determine if BVH node is an inner node or a leaf node by checking the highest bit (bitwise AND operation)
		// inner node if highest bit is 1, leaf node if 0

		if (!(data.x & 0x80000000)) {   // INNER NODE

			// if ray intersects inner node, push indices of left and right child nodes on the stack
			if (RayIntersectsBox(origin, ray, boxIdx)) {
				stack[stackIdx++] = data.y; // right child node index
				stack[stackIdx++] = data.z; // left child node index
				// return if stack size is exceeded
				if (stackIdx>BVH_STACK_SIZE)
				{
					return false; 
				}
			}
		}
		else { // LEAF NODE
			for (unsigned i = data.w; i < data.w + (data.x & 0x7fffffff); i++) {
				// fetch the index of the current triangle
				int idx = tex1Dfetch(g_triIdxListTexture, i).x;
				// check if triangle is the same as the one intersected by previous ray
				// to avoid self-reflections/refractions
				if (avoidSelf == idx)
					continue; 
				// fetch triangle center and normal from texture memory
				float4 center = tex1Dfetch(g_trianglesTexture, 5 * idx);
				float4 normal = tex1Dfetch(g_trianglesTexture, 5 * idx + 1);
				// use the pre-computed triangle intersection data: normal, d, e1/d1, e2/d2, e3/d3
				float k = dot(normal, ray);
				if (k == 0.0f)
					continue; // this triangle is parallel to the ray, ignore it.
				float s = (normal.w - dot(normal, origin)) / k;
				if (s <= 0.0f) // this triangle is "behind" the origin.
					continue;
				if (s <= NUDGE_FACTOR)  // epsilon
					continue;
				Vector3Df hit = ray * s;
				hit += origin;

				// ray triangle intersection
				// Is the intersection of the ray with the triangle's plane INSIDE the triangle?
				float4 ee1 = tex1Dfetch(g_trianglesTexture, 5 * idx + 2);
				float kt1 = dot(ee1, hit) - ee1.w; 
				if (kt1<0.0f) continue;
				float4 ee2 = tex1Dfetch(g_trianglesTexture, 5 * idx + 3);
				float kt2 = dot(ee2, hit) - ee2.w; 
				if (kt2<0.0f) continue;
				float4 ee3 = tex1Dfetch(g_trianglesTexture, 5 * idx + 4);
				float kt3 = dot(ee3, hit) - ee3.w; 
				if (kt3<0.0f) continue;
				// ray intersects triangle, "hit" is the world space coordinate of the intersection.
				{
					// is this intersection closer than all the others?
					float hitZ = distancesq(origin, hit);
					if (hitZ < bestTriDist) {
						// maintain the closest hit
						bestTriDist = hitZ;
						hitdist = sqrtf(bestTriDist);
						pBestTriIdx = idx;
						pointHitInWorldSpace = hit;
						// store barycentric coordinates (for texturing, not used for now)
					}
				}
			}
		}
	}
	
	return pBestTriIdx != -1;
}

template<class T>
__device__ void printv(T &arr, char mark = ' ') {
	printf("%f, %f, %f %c%c%c\n", arr.x, arr.y, arr.z, mark, mark, mark);
}

//////////////////////
// PATH TRACING
//////////////////////
__device__ Vector3Df path_trace(curandState *randstate, Vector3Df rayorig, Vector3Df raydir, int avoidSelf,
	Triangle *pTriangles, int* cudaBVHindexesOrTrilists, float* cudaBVHlimits, float* cudaTriangleIntersectionData, int* cudaTriIdxList)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	// colour mask
	Vector3Df mask = Vector3Df(1.0f, 1.0f, 1.0f);
	// accumulated colour
	for (int bounces = 0; bounces < 6 && mask.lengthsq() > 1e-2; bounces++){  // iteration up to 4 bounces (instead of recursion in CPU code)
		int sphere_id = -1, triangle_id = -1, pBestTriIdx = -1;
		int geomtype = -1;
		const Triangle *pBestTri = NULL;
		Vector3Df neworig, newdir;

		float d = 1e10;
		float scene_t = 1e10;


		float hitdistance = 1e20;
		Vector3Df n, nl; // normal, oriented 

		// intersect all triangles in the scene stored in BVH
		Vector3Df boxnormal = Vector3Df(0, 0, 0);

		BVH_IntersectTriangles(
			cudaBVHindexesOrTrilists, rayorig, raydir, avoidSelf,
			pBestTriIdx, neworig, hitdistance, cudaBVHlimits,
			cudaTriangleIntersectionData, cudaTriIdxList, boxnormal);

		// intersect all spheres in the scene
		int numspheres = sizeof(spheres) / sizeof(Sphere);
		for (int i = numspheres; i--;){  // for all spheres in scene
			// keep track of distance from origin to closest intersection point
			if ((d = spheres[i].intersect(Ray(rayorig, raydir))) && d < scene_t){ 
				scene_t = d; sphere_id = i; geomtype = 1; 
			}
		}
		// set avoidSelf to current triangle index to avoid intersection between this triangle and the next ray, 
		// so that we don't get self-shadow or self-reflection from this triangle...
		avoidSelf = pBestTriIdx;

		if (scene_t >= 1e10) {
			return 0.;
		}

		if (hitdistance < scene_t && hitdistance > 0.002) // EPSILON
		{
			scene_t = hitdistance;
			triangle_id = pBestTriIdx;
			geomtype = 2;
		}
		// SPHERES:
		if (geomtype == 1){

			Sphere &sphere = spheres[sphere_id]; // hit object with closest intersection
			Vector3Df w(0, 1, -.5);
			w.normalize();
			neworig = rayorig + raydir * scene_t;
			neworig.normalize();

			return mask *(exp(4 - 4.f *((w - neworig)).length()));
			/*
			n = neworig - sphere.pos;
			n.normalize();
			nl = dot(n, raydir) < 0 ? n : n * -1;
			// pick two random numbers
			float phi = 2 * M_PI * curand_uniform(randstate);
			float r2 = curand_uniform(randstate);
			float r2s = sqrtf(r2);

			// compute orthonormal coordinate frame uvw with hitpoint as origin 
			Vector3Df u = cross((fabs(nl.x) > .1 ? Vector3Df(0, 1, 0) : Vector3Df(1, 0, 0)), nl); u.normalize();
			Vector3Df v = cross(nl, u);

			// compute cosine weighted random ray direction on hemisphere 
			newdir = u*cosf(phi)*r2s + v*sinf(phi)*r2s + nl*sqrtf(1 - r2);
/*			if (sphere_id == 1 && dot(newdir, neworig) > 0) {
				printf("%f %f %f\n%f %f %f++\n%f\n", neworig.x, neworig.y, neworig.z, nl.x, nl.y, nl.z, dot(newdir, neworig));
			}
			newdir.normalize();
			*/
		}

		// TRIANGLES:5
		if (geomtype == 2){
			pBestTri = &pTriangles[triangle_id];
			// CHECK NORMAL BEFORE EDITING THIS LINE
			n = pBestTri->_normal*1;  // normal
			//n = Vector3Df(0,0,1);
			n.normalize();
	//			printf(dot(n, rayInWorldSpace) < 0 ? "" : "#");
			bool into = dot(n, raydir) < 0;
			nl = into ? n : n * -1;
			
			//Vector3Df colour = pBestTri->_colorf;
#define MU 29.f
#define LAMBDA Vector3Df(.8f, .6f, .2f)
			/*if (x == gridDim.x * blockDim.x / 2 && y == gridDim.y * blockDim.y / 2) {
				printf("%d", into);
			}*/

			if (!into && exp(-MU * hitdistance) < curand_uniform(randstate)) {
				// scattering

				float x1 = raydir.x, x2 = raydir.y, x3 = raydir.z;

				float indic = curand_uniform(randstate),
					phi = curand_uniform(randstate) * 2 * M_PI,
					sin_ind = (1 - indic * indic);

				Vector3Df rand_dir = Vector3Df(cos(phi) * sin_ind, sin(phi) * sin_ind, indic);
				if (abs(x3 - 1) > 1e-5) {
					float denom = sqrt(1 - x3);
					newdir.x = dot(Vector3Df(x1 * x3 / denom, -x2 / denom, x1), rand_dir);
					newdir.y = dot(Vector3Df(x2 * x3 / denom, x1 / denom, x2), rand_dir);
					newdir.z = dot(Vector3Df(-denom, 0, x3), rand_dir);
				}
				else newdir = rand_dir;
				neworig = rayorig - raydir * log((exp(-MU * hitdistance) - 1) * curand_uniform(randstate) + 1) / MU;
				mask *= LAMBDA;
			}
			else {
#define MEDIA_K 1.f  // Index of Refraction air
#define OBJ_K 1.3f  // Index of Refraction glass/water
				float k = into ? MEDIA_K / OBJ_K : OBJ_K / MEDIA_K;  // IOR ratio of refractive materials

				float ddn = dot(raydir, nl);
				float cos2t = 1.0f - k * k * (1.f - ddn*ddn);
				Vector3Df rdir = raydir - n * 2.0f * dot(n, raydir);

				if (cos2t < 0.0f) // total internal reflection 
				{
					newdir = rdir;
					// offset origin next path segment to prevent self intersection
				}
				else // cos2t > 0
				{
					// compute direction of transmission ray
					Vector3Df tdir = raydir * k - nl * (ddn * k + sqrtf(cos2t));
					tdir.normalize();

					float R0 = (OBJ_K - MEDIA_K)*(OBJ_K - MEDIA_K) / (OBJ_K + MEDIA_K)*(OBJ_K + MEDIA_K);
					float c = 1.f - (into ? -ddn : dot(tdir, n));
					float Re = R0 + (1.f - R0) * c * c * c * c * c;

					// randomly choose reflection or transmission ray
					newdir = curand_uniform(randstate) < Re ?
						raydir - n * 2.0f * dot(n, raydir) : tdir;
				}
			}
		}

		// set up origin and direction of next path segment
		
		raydir = newdir;
		raydir.normalize();
		rayorig = neworig + raydir * 1e-4;
	}

	return Vector3Df(0, 0, 0);
}
union Colour  // 4 bytes = 4 chars = 1 float
{
	float c;
	uchar4 components;
};

// the core path tracing kernel, 
// running in parallel for all pixels
__global__ void CoreLoopPathTracingKernel(Vector3Df* output, Vector3Df* accumbuffer, Triangle* pTriangles, Camera* cudaRendercam,
	int* cudaBVHindexesOrTrilists, float* cudaBVHlimits, float* cudaTriangleIntersectionData,
	int* cudaTriIdxList, unsigned int framenumber, unsigned int hashedframenumber)
{

	// assign a CUDA thread to every pixel by using the threadIndex
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	// global threadId, see richiesams blogspot
	int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	// create random number generator and initialise with hashed frame number, see RichieSams blogspot
	curandState randState; // state of the random number generator, to prevent repetition
	curand_init(hashedframenumber + threadId, 0, 0, &randState);

	Vector3Df finalcol; // final pixel colour  
	Vector3Df rendercampos = Vector3Df(cudaRendercam->position.x, cudaRendercam->position.y, cudaRendercam->position.z);


	int i = (height - y - 1)*width + x; // pixel index in buffer
	int pixelx = x; // pixel x-coordinate on screen
	int pixely = height - y - 1; // pixel y-coordintate on screen

	finalcol = Vector3Df(0.0f, 0.0f, 0.0f); // reset colour to zero for every pixel	
	for (int s = 0; s < samps; s++){

		// compute primary ray direction
		// use camera view of current frame (transformed on CPU side) to create local orthonormal basis
		Vector3Df rendercamview = Vector3Df(cudaRendercam->view.x, cudaRendercam->view.y, cudaRendercam->view.z); rendercamview.normalize(); // view is already supposed to be normalized, but normalize it explicitly just in case.
		Vector3Df rendercamup = Vector3Df(cudaRendercam->up.x, cudaRendercam->up.y, cudaRendercam->up.z); rendercamup.normalize();

		Vector3Df horizontalAxis = cross(rendercamview, rendercamup); horizontalAxis.normalize(); // Important to normalize!
		Vector3Df verticalAxis = cross(horizontalAxis, rendercamview); verticalAxis.normalize(); // verticalAxis is normalized by default, but normalize it explicitly just for good measure.

		Vector3Df middle = rendercampos + rendercamview;
		Vector3Df horizontal = horizontalAxis * tanf(cudaRendercam->fov.x * 0.5 * (M_PI / 180)); // Now treating FOV as the full FOV, not half, so I multiplied it by 0.5. I also normzlized A and B, so there's no need to divide by the length of A or B anymore. Also normalized view and removed lengthOfView. Also removed the cast to float.
		Vector3Df vertical = verticalAxis * tanf(-cudaRendercam->fov.y * 0.5 * (M_PI / 180)); // Now treating FOV as the full FOV, not half, so I multiplied it by 0.5. I also normzlized A and B, so there's no need to divide by the length of A or B anymore. Also normalized view and removed lengthOfView. Also removed the cast to float.

		// anti-aliasing
		// calculate center of current pixel and add random number in X and Y dimension
		// based on https://github.com/peterkutz/GPUPathTracer 
		float jitterValueX = curand_uniform(&randState) - 0.5;
		float jitterValueY = curand_uniform(&randState) - 0.5;
		float sx = (jitterValueX + pixelx) / (cudaRendercam->resolution.x - 1);
		float sy = (jitterValueY + pixely) / (cudaRendercam->resolution.y - 1);

		// compute pixel on screen
		Vector3Df pointOnPlaneOneUnitAwayFromEye = middle + ( horizontal * ((2 * sx) - 1)) + ( vertical * ((2 * sy) - 1));
		Vector3Df pointOnImagePlane = rendercampos + ((pointOnPlaneOneUnitAwayFromEye - rendercampos) * cudaRendercam->focalDistance); // Important for depth of field!		

		// calculation of depth of field / camera aperture 
		// based on https://github.com/peterkutz/GPUPathTracer 
		
		Vector3Df aperturePoint;

		if (cudaRendercam->apertureRadius > 0.00001) { // the small number is an epsilon value.
		
			// generate random numbers for sampling a point on the aperture
			float random1 = curand_uniform(&randState);
			float random2 = curand_uniform(&randState);

			// randomly pick a point on the circular aperture
			float angle = TWO_PI * random1;
			float distance = cudaRendercam->apertureRadius * sqrtf(random2);
			float apertureX = cos(angle) * distance;
			float apertureY = sin(angle) * distance;

			aperturePoint = rendercampos + (horizontalAxis * apertureX) + (verticalAxis * apertureY);
		}
		else { // zero aperture
			aperturePoint = rendercampos;
		}

		// calculate ray direction of next ray in path
		Vector3Df apertureToImagePlane = pointOnImagePlane - aperturePoint; 
		apertureToImagePlane.normalize(); // ray direction, needs to be normalised
		Vector3Df rayInWorldSpace = apertureToImagePlane;
		// in theory, this should not be required
		rayInWorldSpace.normalize();

		// origin of next ray in path
		Vector3Df originInWorldSpace = aperturePoint;

		finalcol += path_trace(&randState, originInWorldSpace, rayInWorldSpace, -1, pTriangles, 
			cudaBVHindexesOrTrilists, cudaBVHlimits, cudaTriangleIntersectionData, cudaTriIdxList) * (1.0f/samps);
	}       

	// add pixel colour to accumulation buffer (accumulates all samples) 
	accumbuffer[i] += finalcol;
	// averaged colour: divide colour by the number of calculated frames so far
	Vector3Df tempcol = accumbuffer[i] / framenumber;

	Colour fcolour;
	Vector3Df colour = Vector3Df(clamp(tempcol.x, 0.0f, 1.0f), clamp(tempcol.y, 0.0f, 1.0f), clamp(tempcol.z, 0.0f, 1.0f));
	// convert from 96-bit to 24-bit colour + perform gamma correction
	fcolour.components = make_uchar4((unsigned char)(powf(colour.x, 1 / 2.2f) * 255), (unsigned char)(powf(colour.y, 1 / 2.2f) * 255), (unsigned char)(powf(colour.z, 1 / 2.2f) * 255), 1);
	// store pixel coordinates and pixelcolour in OpenGL readable outputbuffer
	output[i] = Vector3Df(x, y, fcolour.c);

}

bool g_bFirstTime = true;

// the gateway to CUDA, called from C++ (in void disp() in main.cpp)
void cudarender(Vector3Df* dptr, Vector3Df* accumulatebuffer, Triangle* cudaTriangles, int* cudaBVHindexesOrTrilists,
	float* cudaBVHlimits, float* cudaTriangleIntersectionData, int* cudaTriIdxList, 
	unsigned framenumber, unsigned hashedframes, Camera* cudaRendercam){

	if (g_bFirstTime) {
		// if this is the first time cudarender() is called,
		// bind the scene data to CUDA textures!
		g_bFirstTime = false;

		printf("g_triIndexListNo: %d\n", g_triIndexListNo);
		printf("g_pCFBVH_No: %d\n", g_pCFBVH_No);
		printf("g_verticesNo: %d\n", g_verticesNo);
		printf("g_trianglesNo: %d\n", g_trianglesNo);

		cudaChannelFormatDesc channel1desc = cudaCreateChannelDesc<uint1>();
		cudaBindTexture(NULL, &g_triIdxListTexture, cudaTriIdxList, &channel1desc, g_triIndexListNo * sizeof(uint1));

		cudaChannelFormatDesc channel2desc = cudaCreateChannelDesc<float2>();
		cudaBindTexture(NULL, &g_pCFBVHlimitsTexture, cudaBVHlimits, &channel2desc, g_pCFBVH_No * 6 * sizeof(float));

		cudaChannelFormatDesc channel3desc = cudaCreateChannelDesc<uint4>();
		cudaBindTexture(NULL, &g_pCFBVHindexesOrTrilistsTexture, cudaBVHindexesOrTrilists, &channel3desc,
			g_pCFBVH_No * sizeof(uint4));

		//cudaChannelFormatDesc channel4desc = cudaCreateChannelDesc<float4>();
		//cudaBindTexture(NULL, &g_verticesTexture, cudaPtrVertices, &channel4desc, g_verticesNo * 8 * sizeof(float));

		cudaChannelFormatDesc channel5desc = cudaCreateChannelDesc<float4>();
		cudaBindTexture(NULL, &g_trianglesTexture, cudaTriangleIntersectionData, &channel5desc, g_trianglesNo * 20 * sizeof(float));
	}
	dim3 block(32, 32, 1);   // dim3 CUDA specific syntax, block and grid are required to schedule CUDA threads over streaming multiprocessors
	dim3 grid(width / block.x, height / block.y, 1);

	/*cudaEvent_t     start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);*/

	CoreLoopPathTracingKernel << <grid, block >> >(dptr, accumulatebuffer, cudaTriangles, cudaRendercam, cudaBVHindexesOrTrilists,
		cudaBVHlimits, cudaTriangleIntersectionData, cudaTriIdxList, framenumber, hashedframes);
	// get stop time, and display the timing results
	/*cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float   elapsedTime;
	cudaEventElapsedTime(&elapsedTime,
		start, stop);
	printf("Time to generate:  %3.1f ms\n", elapsedTime);*/
}
