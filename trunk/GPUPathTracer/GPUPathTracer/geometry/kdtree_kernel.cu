#include "KDTree.h"
#include "utils.h"
//得到一条光线与一个三角形是否相交
__device__  inline float Triangle_getDistanceUntilHit(Triangle *MyTriangle,const Ray& ray,float& U,float& V) 
{

	//Vec3 A=vecSub(MyTriangle->v2,MyTriangle->v1),B=vecSub(MyTriangle->v3,MyTriangle->v1),C=vecMul(ray.m_unitDir,-1.0f),D=vecSub(ray.m_startPos,MyTriangle->v1);
	Vec3 A=MyTriangle->v2-MyTriangle->v1,B=MyTriangle->v3-MyTriangle->v1,C=ray.m_unitDir*-1.0f,D=ray.m_startPos-MyTriangle->v1;
	float a=A.x,b=B.x,c=C.x,
		d=A.y,e=B.y,f=C.y,
		g=A.z,h=B.z,i=C.z,
		j=D.x,k=D.y,l=D.z;
	float det,det0,det1,det2;
	float T=FLT_MAX;
	det=a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g);
	det0=j*(e*i-f*h)-b*(k*i-f*l)+c*(k*h-e*l);
	det1=a*(k*i-f*l)-j*(d*i-f*g)+c*(d*l-k*g);
	det2=a*(e*l-k*h)-b*(d*l-k*g)+j*(d*h-e*g);
	U=det0/det;
	V=det1/det;
	T=det2/det;

	//判断条件放宽以减轻边缘走样
	if(T>0.000001f&&-0.000001f<=U&&-0.000001f<=V&&U+V<=1.000001f)return T;
	//if(T>0&&0<=U&&U<=1&&0<=V&&V<=1&&0<=U+V&&U+V<=1)return T;
	else return -666.0f;/**/
}

//dev_RayTriIntersect: triangle intersection test
//a/b/c are vertices of triangle
//o is ray origin, d is ray direction
//out_lambda is intersection parameter lambda, out_bary1/2 are barycentric hit coordinates
//adapted from MNRT code by Mathias Neumann
inline __device__ bool FindTriangleIntersect(const float3 a, const float3 b, const float3 c, 
											 const float3 o, const float3 d,
											 float& out_lambda, float& out_bary1, float& out_bary2)
{
	float3 edge1 = b - a;
	float3 edge2 = c - a;

	float3 pvec = cross(d, edge2);
	float det = dot(edge1, pvec);
	if(det == 0.f)
		return false;
	float inv_det = 1.0f / det;

	float3 tvec = o - a;
	out_bary1 = dot(tvec, pvec) * inv_det;

	float3 qvec = cross(tvec, edge1);
	out_bary2 = dot(d, qvec) * inv_det;
	out_lambda = dot(edge2, qvec) * inv_det;

	bool hit = (out_bary1 >= 0.0f && out_bary2 >= 0.0f && (out_bary1 + out_bary2) <= 1.0f);
	return hit;
}
//检查一条光线与一个包围盒是否有交点
__device__ bool insectWithBox(const Ray& ray,Vec3 BL,Vec3 BH)
{
	float tnear=-FLT_MAX,tfar=FLT_MAX;
	float t1,t2;
	//return true;
	for(int i=0;i<3;i++)
	{
		if(ray.m_unitDir[i]==0.0f&&(ray.m_startPos[i]>BH[i]||ray.m_startPos[i]<BL[i]))
			return false;
		t1 = (BL[i]-ray.m_startPos[i])/ray.m_unitDir[i];
		t2 = (BH[i]-ray.m_startPos[i])/ray.m_unitDir[i];
		if (t1 > t2)
		{
			float temp=t2;
			t2=t1;
			t1=temp;
		}
		if (t1 > tnear)
			tnear = t1;
		if (tfar > t2)
			tfar = t2;
		if (tnear - tfar > PRECISION||tfar < 0.0f) 
			return false;
	}
	return true;
}
//查找KDTree，找到与光线相交的最近的三角形
__device__   void RayInterWithCube(const int EnterFace ,int &face,TYPE &dist,TYPE finalInterPoint[3],Ray ray,TYPE * tMinCoordinate,TYPE *tMaxCoordinate)
{
	dist=INF;
	face=-10;
	int key=0;
	TYPE interPoint[3];
	for(int i=0;i<3;i++)
	{
		TYPE tempStart[3]={ray.m_startPos.x,ray.m_startPos.y,ray.m_startPos.z},tempUnit[3]={ray.m_unitDir.x,ray.m_unitDir.y,ray.m_unitDir.z};
		if(fabs(tempUnit[i])>1e-06&&(tMinCoordinate[i]-tempStart[i])/tempUnit[i]>=0)
		{
			TYPE tempDist=0;
			int j;
			for(j=0;j<3;j++)
			{
				if(i==j)
					interPoint[j]=tMinCoordinate[j];
				else
					interPoint[j]=tempStart[j]+(tMinCoordinate[i]-tempStart[i])*(tempUnit[j]/tempUnit[i]);
				if(tMinCoordinate[j]-interPoint[j]>PRECISION||interPoint[j]-tMaxCoordinate[j]>PRECISION) break;//为了确保精度，保留小数点后5位精确
				tempDist+=(interPoint[j]-tempStart[j])*(interPoint[j]-tempStart[j]);
			}
			if(j==3)
			{
				tempDist=sqrt(tempDist);
				if(tempDist<dist&&(abs(2*i-EnterFace)!=1||EnterFace/2!=i)) 
				{
					if(fabs(tempDist)<PRECISION) 
					{
						if(fabs(dist-INF)<PRECISION)
						{
							key=1;
							face=2*i;
							memcpy(finalInterPoint,interPoint,sizeof(TYPE)*3);
						}
					}
					else
					{
						dist=tempDist;
						face=2*i;
						memcpy(finalInterPoint,interPoint,sizeof(TYPE)*3);
					}
				}
			}
		}
		if(fabs(tempUnit[i])>1e-06&&(tMaxCoordinate[i]-tempStart[i])/tempUnit[i]>=0)
		{
			TYPE tempDist=0;
			int j;
			for(j=0;j<3;j++)
			{
				if(i==j)
					interPoint[j]=tMaxCoordinate[j];
				else
					interPoint[j]=tempStart[j]+(tMaxCoordinate[i]-tempStart[i])*(tempUnit[j]/tempUnit[i]);
				if(tMinCoordinate[j]-interPoint[j]>PRECISION||interPoint[j]-tMaxCoordinate[j]>PRECISION) break;//为了确保精度，保留小数点后5位精确
				tempDist+=(interPoint[j]-tempStart[j])*(interPoint[j]-tempStart[j]);
			}
			if(j==3)
			{
				tempDist=sqrt(tempDist);
				if(tempDist<dist&&(abs(2*i+1-EnterFace)!=1||EnterFace/2!=(2*i+1)/2)) 
				{	 
					if(fabs(tempDist)<PRECISION) 
					{
						if(fabs(dist-INF)<PRECISION)
						{
							key=1;
							face=2*i+1;
							memcpy(finalInterPoint,interPoint,sizeof(TYPE)*3);
						}
					}
					else
					{
						dist=tempDist;
						face=2*i+1;
						memcpy(finalInterPoint,interPoint,sizeof(TYPE)*3);
					}
				}
			}
		}
	}
	if(key&&dist==INF)dist=0.0f;

}

__device__   void RopeLocalRayNearestTriangle(Knode *TreeNode, 
											  int *TriangleIndexUseForCuda, Triangle *treeTriangle, 
											  int &TriangleIndex, float &MinDist, Ray ray,int index,float &U,float &V)
{
	TYPE EnterDist=0,LeaveDist=INF,TotDist=0;
	TriangleIndex=-1;
	TYPE point[3]={ray.m_startPos.x,ray.m_startPos.y,ray.m_startPos.z};
	Vec3 iniPos(ray.m_startPos);
	MinDist=INF;
	int ZeroCount=0;
	int face=-10,nextFace;    
	if(iniPos.x<TreeNode[index].MinCoordinate[0]||iniPos.x>TreeNode[index].MaxCoordinate[0]||
		iniPos.y<TreeNode[index].MinCoordinate[1]||iniPos.y>TreeNode[index].MaxCoordinate[1]||
		iniPos.z<TreeNode[index].MinCoordinate[2]||iniPos.z>TreeNode[index].MaxCoordinate[2])
	{

		RayInterWithCube(-5,face,EnterDist,point,ray,TreeNode[index].MinCoordinate,TreeNode[index].MaxCoordinate);
		TotDist+=EnterDist;
		if(face%2) face--;
		else face++;
	}

	while(EnterDist<LeaveDist)
	{
		Vec3 Temp(point[0],point[1],point[2]);
		float Node[3]={Temp.x,Temp.y,Temp.z};
		while(TreeNode[index].leftChild>=0&&TreeNode[index].rightChild>=0)//如果该结点不是叶子节点
		{
			int axis=TreeNode[index].planeAxis;
			if(Node[axis]<TreeNode[index].splitPlane)
				index=TreeNode[index].leftChild;
			else index=TreeNode[index].rightChild;
		}
		ray.m_startPos=Temp;
		bool IfContinue=true;
		float tempU,tempV;
		for(int i=0;i<TreeNode[index].TriangleNumber;i++)
		{
			Triangle *p=&treeTriangle[TriangleIndexUseForCuda[TreeNode[index].CudaOffset+i]];
			Vec3 tmpV(ray.m_startPos);
			ray.m_startPos=iniPos;
			TYPE distance;
			//FindTriangleIntersect(p->v1.toFloat3(),p->v2.toFloat3(),p->v3.toFloat3(),ray.m_startPos.toFloat3(),ray.m_unitDir.toFloat3(),distance,tempU,tempV);
			distance=Triangle_getDistanceUntilHit(p,ray,tempU,tempV);
			ray.m_startPos=tmpV;
			if(distance>ray.error_offset && distance < LeaveDist && distance < ray.segment)
			{ 
				LeaveDist=distance;
				U=tempU;
				V=tempV;
				TriangleIndex=TriangleIndexUseForCuda[TreeNode[index].CudaOffset+i];
			}
		}
		//if(!IfContinue) break;
		//if(fabs(LeaveDist-INF)>PRECISION) break;
		RayInterWithCube(face,nextFace,EnterDist,point,ray,TreeNode[index].MinCoordinate,TreeNode[index].MaxCoordinate);
		if(nextFace==-10)//不要问我为什么设为-10。。其实是随便设的，只是表示无面出
			break;
		index=TreeNode[index].RopeIndex[nextFace];
		if(index==-1) break;
		if(fabs(EnterDist)<PRECISION)
			ZeroCount++;
		else ZeroCount=0;
		if(ZeroCount>=2) return;
		TotDist+=EnterDist;
		EnterDist=TotDist;
		face=nextFace;
	}
	MinDist=LeaveDist;
}
__global__  void active_ray_bunch_traverse_kernel(Knode *TreeNode, int *TriangleIndexUseForCuda, Triangle *treeTriangle,
												 Ray *ray,int *activePixels,int numActivePixels,int *hitID,float *lamda,float *U,float *V){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if(tid >= numActivePixels)
		return;
	int uid =  activePixels[tid];
	if(uid < 0)
		printf("%d\n",uid);
	RopeLocalRayNearestTriangle(TreeNode,TriangleIndexUseForCuda,treeTriangle,hitID[uid],lamda[uid],ray[uid],0,U[uid],V[uid]);

}
__global__ void active_ray_bunch_traverse_point_kernel(Knode *TreeNode, int *TriangleIndexUseForCuda, Triangle *treeTriangle,
												       Ray *ray,int *activePixels,int numActivePixels,int *hitsID){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if(tid >= numActivePixels)
		return;
	int uid =  activePixels[tid];
	int hitID;
	float lamda,U,V;
	Ray curRay = ray[uid];
	RopeLocalRayNearestTriangle(TreeNode,TriangleIndexUseForCuda,treeTriangle,hitID,lamda,curRay,0,U,V);
	hitsID[uid] = hitID;
}
extern "C" void active_ray_bunch_traverse(Knode *TreeNode, int *TriangleIndexUseForCuda, Triangle *treeTriangle, 
										 DevVector<Ray> &ray,DevVector<int> &activePixels,
										 DevVector<int> &hitID , DevVector<float> &lamda, DevVector<float> &U,DevVector<float> &V){
   dim3 block(256,1,1);
   dim3 grid(IntegerDivide(256)(activePixels.size()),1,1);
   cudaThreadSynchronize();
   active_ray_bunch_traverse_kernel<<<grid,block>>>(TreeNode,TriangleIndexUseForCuda,treeTriangle,ray.pointer(),
	                                               activePixels.pointer(),activePixels.size(),hitID.pointer(),
												   lamda.pointer(),U.pointer(),V.pointer());
   cudaThreadSynchronize();
   CUT_CHECK_ERROR("active_ray_bunch_travese_kernel fail!");
}
extern "C" void active_ray_bunch_traverse_hits(Knode *TreeNode, int *TriangleIndexUseForCuda, Triangle *treeTriangle, 
												DevVector<Ray> &ray,DevVector<int> &activePixels,
												DevVector<int> &hitsID){
   dim3 block(256,1,1);
   dim3 grid(IntegerDivide(256)(activePixels.size()),1,1);
   active_ray_bunch_traverse_point_kernel<<<grid,block>>>(TreeNode,TriangleIndexUseForCuda,treeTriangle,ray.pointer(),
														  activePixels.pointer(),activePixels.size(),hitsID.pointer());
}