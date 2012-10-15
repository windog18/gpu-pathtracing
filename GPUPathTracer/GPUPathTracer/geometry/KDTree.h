#pragma once
#include "ray.h"
#include "triangle.h"
#include<vector>
#include<map>
#include<algorithm>
#define TYPE float
#define DIMENS  3
#define INF 1e08
#define PRECISION 1e-07
#define TRAVERSAL_TIME 15.0f
#define INTERSECTION_TIME 20.0f
#define SWAP(TT,A,B)  {Knode *tmp=TT[A]; TT[A]=TT[B]; TT[B]=tmp;}
using namespace std;
typedef struct 
{
    TYPE coord[DIMENS];
	int planeAxis;
	TYPE splitPlane;
	int Pid;
	int TriangleNumber;
	int CudaOffset;
	int *trianglePtr;
	TYPE MinCoordinate[DIMENS];
	TYPE MaxCoordinate[DIMENS];
	int leftChild;
	int rightChild;
	int RopeIndex[6];//left, right, top,bottom, front,back
}Knode;
typedef struct
{
	int planeAxis;
    double splitPlane;
	int triangleIndex;
	int triangleDirection;//0 代表 -，1代表|,2代表+
}TriangleEvent;
typedef struct
{
	TYPE MinCoordinate[DIMENS];
	TYPE MaxCoordinate[DIMENS];
}TriangleVoxel;
typedef struct
{
	int index;
	int HaveVisit;
}StackNode;

///这个程序的数组从0开始使用
class KdTree
{
	public:
	 KdTree(int NodeNumber,int TriangleNumber);
	 ~KdTree();
	 void BlanceTree();  
	 void UseSAHToCreateTree();
	 void transformTreeToGPU();
	 void Store(TYPE CoordX,TYPE CoordY,TYPE CoordZ,int Pid);
	 void Store(TYPE Coord[DIMENS],int Pid);
	 void RopeLocalRayNearestTriangle(Knode *TreeNode,int NodeNumber
										  ,int *TriangleIndexUseForCuda,Triangle * treeTriangle,
										  float *MinCoordinate,float *MaxCoordinate,
										  int &TriangleIndex,float &MinDist,Ray ray,int index);//求光线R最近碰到的三角面块
	 void AddTriangleToScene(Triangle *p);///添加三角面片到场景中 
	 void destoryTree();
	 Knode * TreeNode, *TreeNode_device;
	 TYPE MaxCoordinate[DIMENS];
	 TYPE MinCoordinate[DIMENS];
	 int NodeNumber;
	 int TriangleNumber;
	 Triangle *treeTriangle, *treeTriangle_device;  
	 int *TriangleIndexUseForCuda, *TriangleIndexUseForCuda_device;//一维数组存所有节点的面片索引信息
	 int TriangleIndexCount;//上面一维数组的总长度
	 int MaxStack;
	 private:
	//用SAH方法寻找切割点
	 float SAHFindPlane(int nodeIndex,vector<TriangleEvent> &pEvent,int &nPSide, int &planeAxis, float &splitPlane, int &leftN, int &rightN, int &middleN);
	 int GetAvailableNodeIndex()
	 {
		   if(availableNode<16*TriangleNumber)
			return availableNode++;
		   else return -1;
	 }
     void SAHBuildTree(int index,vector<TriangleEvent> &pEvent,TYPE fatherCost,int depth);
	 float SAHcost(int &NpToside,int axis,TYPE axisPlane,TYPE *MinCoordinate,TYPE *MaxCoordinate,int nL,int nR,int nP);
	 void SplitTriangle(int triangleIndex,TYPE *MinCoordinate,TYPE *MaxCoordinate,vector<TriangleEvent> *iniEvent);
	 void ConvertIndexToLinear();///为了方便CUDA使用，将所有节点中三角面片索引信息转化为一个一维数组，并把每个节点的索引起点赋值到结点信息中
	

	 /// RopeFind Function
	 void RopeConstruction(int index,int *FatherRope,TYPE *tMinCoordinate,TYPE *tMaxCoordinate);
	 void RopeOptimize(int &index,int splitFace,TYPE *tMinCoordinate,TYPE *tMaxCoordinate);
	 void RayInterWithCube(const int EnterFace ,int &face,TYPE &dist,TYPE finalInterPoint[3],Ray ray,TYPE * tMinCoordinate,TYPE *tMaxCoordinate);	
	 ///
	 void PushStack(StackNode *stack,StackNode p,int &top);
	 StackNode PopStack(StackNode *stack,int &top);
	 StackNode GetTopNode(StackNode *stack,int &top);	
	 int availableNode;
	 int HaveNode;
	 int HaveTriangle;
	 int MaxDepth;
     map<int,int> myDir;
};