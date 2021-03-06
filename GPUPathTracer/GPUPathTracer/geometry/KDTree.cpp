
#include"KDTree.h"
#include"math.h"
bool   lessmark(const   TriangleEvent&   s1,const   TriangleEvent&   s2)   
{   
	if(s1.planeAxis!=s2.planeAxis) return s1.planeAxis<s2.planeAxis;
	else if(s1.planeAxis==s2.planeAxis&&fabs(s1.splitPlane-s2.splitPlane)>PRECISION) return s1.splitPlane<s2.splitPlane;
	else if(s1.planeAxis==s2.planeAxis&&fabs(s1.splitPlane-s2.splitPlane)<PRECISION) return s1.triangleDirection<s2.triangleDirection;
	else return s1.planeAxis<s2.planeAxis;
} 

KdTree::KdTree(int NodeNumber,int TriangleNumber)
{
	HaveNode=0;
	HaveTriangle=0;
	MaxStack=0;
	availableNode=1;
	TreeNode=new Knode[16*TriangleNumber];
	treeTriangle=new Triangle[TriangleNumber];
	this->NodeNumber=NodeNumber;
	this->TriangleNumber=TriangleNumber;
	for(int i=0;i<DIMENS;i++)
	{
		MaxCoordinate[i]=-INF;
		MinCoordinate[i]=INF;
	}
	for(int i=0;i<NodeNumber;i++)
		TreeNode[i].TriangleNumber=0;
	MaxDepth=1.2*log(TriangleNumber*1.0f)/log(2.0)+2;
	TreeNode[0].TriangleNumber=this->TriangleNumber;
	TreeNode[0].trianglePtr=new int[this->TriangleNumber];
	for(int i=0;i<TreeNode[0].TriangleNumber;i++)
		TreeNode[0].trianglePtr[i]=i;
}
KdTree::~KdTree()
{
	free(TreeNode);
}
void KdTree::Store(TYPE Coord[DIMENS],int Pid)
{
	Knode tmp;
	memcpy(tmp.coord,Coord,sizeof(TYPE)*DIMENS);
	tmp.Pid=Pid;
	TreeNode[HaveNode++]=tmp;
	for(int i=0;i<DIMENS;i++)
	{
		if(tmp.coord[i]>MaxCoordinate[i])
			MaxCoordinate[i]=tmp.coord[i];
		if(tmp.coord[i]<MinCoordinate[i])
			MinCoordinate[i]=tmp.coord[i];
	}
}
void KdTree::Store(TYPE CoordX, TYPE CoordY, TYPE CoordZ, int Pid)
{
	TYPE Coord[4]={CoordX,CoordY,CoordZ};
	Store(Coord,Pid);
}

void KdTree::PushStack(StackNode *stack, StackNode p, int &top)
{
	stack[top++]=p;
}
StackNode KdTree::PopStack(StackNode *stack, int &top)
{
	 return stack[--top]; 
}
StackNode KdTree::GetTopNode(StackNode *stack, int &top)
{
	return stack[top-1];
}
void KdTree::AddTriangleToScene(Triangle *p)
{
	float v[3][3]={{p->v1.x,p->v1.y,p->v1.z},{p->v2.x,p->v2.y,p->v2.z},{p->v3.x,p->v3.y,p->v3.z}};
	for(int i=0;i<3;i++)
		for(int j=0;j<3;j++)
		{
			if(v[i][j]<MinCoordinate[i]) MinCoordinate[i]=v[i][j];
			if(v[i][j]>MaxCoordinate[i]) MaxCoordinate[i]=v[i][j];
	    }
   memcpy(&treeTriangle[HaveTriangle++],p,sizeof(Triangle));
}
void KdTree::ConvertIndexToLinear()
{
		int totCount=0;
		for(int i=0;i<NodeNumber;i++) 
		{
			TreeNode[i].CudaOffset=totCount;
			totCount+=TreeNode[i].TriangleNumber;
		}
        TriangleIndexUseForCuda=new int[totCount];
		int Tcount=0;
		for(int i=0;i<NodeNumber;i++)
		{
			for(int j=0;j<TreeNode[i].TriangleNumber;j++)
			{
				TriangleIndexUseForCuda[Tcount++]=TreeNode[i].trianglePtr[j];
			}
		}
		TriangleIndexCount=totCount;
}

void KdTree::RopeOptimize(int &index, int splitFace, float *tMinCoordinate, float *tMaxCoordinate)
{
	if(index<0) return ;
	while(TreeNode[index].leftChild>=0&&TreeNode[index].rightChild>=0)//如果不是叶子节点
	{
		int splitAxis=TreeNode[index].planeAxis;
		if(splitAxis==splitFace/2)//如果结点的ROPE结点分界面与结点平行
		{
			if(splitFace-1>=0&&(splitFace-1)/2==splitAxis)
				index=TreeNode[index].leftChild;
			else
				index=TreeNode[index].rightChild;
		}
		else if(TreeNode[index].splitPlane<=tMinCoordinate[splitAxis])
			index=TreeNode[index].rightChild;
		else if(TreeNode[index].splitPlane>=tMaxCoordinate[splitAxis]) 
			index=TreeNode[index].leftChild;
		else break;
	}
}
void KdTree::RopeConstruction(int index, int *FatherRope, TYPE *tMinCoordinate, TYPE *tMaxCoordinate)
{
	memcpy(TreeNode[index].MinCoordinate,tMinCoordinate,sizeof(TYPE)*DIMENS);
	memcpy(TreeNode[index].MaxCoordinate,tMaxCoordinate,sizeof(TYPE)*DIMENS);
	if(TreeNode[index].leftChild<0||TreeNode[index].rightChild<0)//如果这个结点为叶子节点
		return;
	for(int i=0;i<6;i++)  
	  RopeOptimize(TreeNode[index].RopeIndex[i],i,tMinCoordinate,tMaxCoordinate);
   TYPE tmp;
   int splitAxis=TreeNode[index].planeAxis;
   ///递归处理左子结点
   memcpy(TreeNode[TreeNode[index].leftChild].RopeIndex,TreeNode[index].RopeIndex,sizeof(int)*6);
   TreeNode[TreeNode[index].leftChild].RopeIndex[splitAxis*2+1]=TreeNode[index].rightChild;
   tmp=tMaxCoordinate[splitAxis];
   tMaxCoordinate[splitAxis]=TreeNode[index].splitPlane;
   RopeConstruction(TreeNode[index].leftChild,TreeNode[index].RopeIndex,tMinCoordinate,tMaxCoordinate);
   tMaxCoordinate[splitAxis]=tmp;
   ///递归处理右子结点
   memcpy(TreeNode[TreeNode[index].rightChild].RopeIndex,TreeNode[index].RopeIndex,sizeof(int)*6);
   TreeNode[TreeNode[index].rightChild].RopeIndex[splitAxis*2]=TreeNode[index].leftChild;
   tmp=tMinCoordinate[splitAxis];
   tMinCoordinate[splitAxis]=TreeNode[index].splitPlane;
   RopeConstruction(TreeNode[index].rightChild,TreeNode[index].RopeIndex,tMinCoordinate,tMaxCoordinate);
   tMinCoordinate[splitAxis]=tmp;
}
void KdTree::RayInterWithCube(const int EnterFace,int &face, float &dist,TYPE finalInterPoint[3] ,Ray ray, TYPE *tMinCoordinate, TYPE *tMaxCoordinate)
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
void KdTree::RopeLocalRayNearestTriangle(Knode *TreeNode, int NodeNumber, 
										 int *TriangleIndexUseForCuda, Triangle *treeTriangle, 
										 float *MinCoordinate, float *MaxCoordinate,
										 int &TriangleIndex, float &MinDist, Ray ray,int index)
{
		TYPE EnterDist=0,LeaveDist=INF,TotDist=0;
		TriangleIndex=-1;
		TYPE point[3]={ray.m_startPos.x,ray.m_startPos.y,ray.m_startPos.z};
		Vec3 iniPos(ray.m_startPos);
		MinDist=INF;
		int ZeroCount=0;
		int face=-10,nextFace;
		if(iniPos.x<TreeNode[0].MinCoordinate[0]||iniPos.x>TreeNode[0].MaxCoordinate[0]||
		   iniPos.y<TreeNode[0].MinCoordinate[1]||iniPos.y>TreeNode[0].MaxCoordinate[1]||
		   iniPos.z<TreeNode[0].MinCoordinate[2]||iniPos.z>TreeNode[0].MaxCoordinate[2])
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
			for(int i=0;i<TreeNode[index].TriangleNumber;i++)
			{
				Triangle *p=&treeTriangle[TriangleIndexUseForCuda[TreeNode[index].CudaOffset+i]];
				Vec3 tmpV(ray.m_startPos);
				ray.m_startPos=iniPos;
				TYPE distance=p->getDistanceUntilHit(ray);
				ray.m_startPos=tmpV;
				if(distance>=0&&distance<LeaveDist)
				{ 
					LeaveDist=distance;
					TriangleIndex=TriangleIndexUseForCuda[TreeNode[index].CudaOffset+i];
				}
			}
			//if(!IfContinue) break;
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
float KdTree::SAHcost(int &NpToside,int axis, TYPE axisPlane, TYPE *MinCoordinate,TYPE *MaxCoordinate, int nL, int nR, int nP)
{
	TYPE totSur=0,leftSur=0,rightSur=0;
	for(int i=0;i<3;i++)
	{
		TYPE temp=(MaxCoordinate[i]-MinCoordinate[i])*(MaxCoordinate[(i+1)%3]-MinCoordinate[(i+1)%3]);
		totSur+=temp;
		if(i==axis)
		{
			leftSur+=(axisPlane-MinCoordinate[i])*(MaxCoordinate[(i+1)%3]-MinCoordinate[(i+1)%3]);
			rightSur+=(MaxCoordinate[i]-axisPlane)*(MaxCoordinate[(i+1)%3]-MinCoordinate[(i+1)%3]);
		}
		else if((i+1)%3==axis)
		{  
			leftSur+=(MaxCoordinate[i]-MinCoordinate[i])*(axisPlane-MinCoordinate[(i+1)%3]);
			rightSur+=(MaxCoordinate[i]-MinCoordinate[i])*(MaxCoordinate[(i+1)%3]-axisPlane);
		}
		else
		{
			leftSur+=temp;
			rightSur+=temp;
		}
	}
	
    TYPE leftCount=TRAVERSAL_TIME+INTERSECTION_TIME*(leftSur/totSur*(nL+nP)+rightSur/totSur*nR);
	if(nL+nP==0||nR==0) leftCount*=0.8f;
	TYPE rightCount=TRAVERSAL_TIME+INTERSECTION_TIME*(leftSur/totSur*(nL)+rightSur/totSur*(nR+nP));
	if(nR+nP==0||nL==0) rightCount*=0.8f;
	if(leftCount<rightCount)
	{
		NpToside=0;
		return leftCount;
	}
	else
	{
		NpToside=1;
		return rightCount;
	}
}

void KdTree::SplitTriangle(int triangleIndex,TYPE *MinCoordinate,TYPE *MaxCoordinate,vector<TriangleEvent> *iniEvent)
{
	//spit Triangle into volex
		TYPE *GetMinCoordinate, *GetMaxCoordinate;
	Triangle *pTriangle=&treeTriangle[triangleIndex];
	GetMinCoordinate=new TYPE[DIMENS];
	GetMaxCoordinate=new TYPE[DIMENS];
    GetMinCoordinate[0]=pTriangle->v1.x;
	GetMaxCoordinate[0]=pTriangle->v1.x;
	GetMinCoordinate[1]=pTriangle->v1.y;
	GetMaxCoordinate[1]=pTriangle->v1.y;
	GetMinCoordinate[2]=pTriangle->v1.z;
	GetMaxCoordinate[2]=pTriangle->v1.z;

    if(pTriangle->v2.x<GetMinCoordinate[0]) GetMinCoordinate[0]=pTriangle->v2.x;
	if(pTriangle->v2.x>GetMaxCoordinate[0]) GetMaxCoordinate[0]=pTriangle->v2.x;
	if(pTriangle->v2.y<GetMinCoordinate[1]) GetMinCoordinate[1]=pTriangle->v2.y;
	if(pTriangle->v2.y>GetMaxCoordinate[1]) GetMaxCoordinate[1]=pTriangle->v2.y;
	if(pTriangle->v2.z<GetMinCoordinate[2]) GetMinCoordinate[2]=pTriangle->v2.z;
	if(pTriangle->v2.z>GetMaxCoordinate[2]) GetMaxCoordinate[2]=pTriangle->v2.z;

	if(pTriangle->v3.x<GetMinCoordinate[0]) GetMinCoordinate[0]=pTriangle->v3.x;
	if(pTriangle->v3.x>GetMaxCoordinate[0]) GetMaxCoordinate[0]=pTriangle->v3.x;
	if(pTriangle->v3.y<GetMinCoordinate[1]) GetMinCoordinate[1]=pTriangle->v3.y;
	if(pTriangle->v3.y>GetMaxCoordinate[1]) GetMaxCoordinate[1]=pTriangle->v3.y;
	if(pTriangle->v3.z<GetMinCoordinate[2]) GetMinCoordinate[2]=pTriangle->v3.z;
	if(pTriangle->v3.z>GetMaxCoordinate[2]) GetMaxCoordinate[2]=pTriangle->v3.z;
    for(int i=0;i<DIMENS;i++)
	{
		if(GetMinCoordinate[i]<MinCoordinate[i]) GetMinCoordinate[i]=MinCoordinate[i];
		if(GetMaxCoordinate[i]>MaxCoordinate[i]) GetMaxCoordinate[i]=MaxCoordinate[i];
	}

		for(int j=0;j<DIMENS;j++)
		{
			float v1[3]={pTriangle->v1.x,pTriangle->v1.y,pTriangle->v1.z};
		    float v2[3]={pTriangle->v2.x,pTriangle->v2.y,pTriangle->v2.z};
			float v3[3]={pTriangle->v3.x,pTriangle->v3.y,pTriangle->v3.z};    
			TriangleEvent tempEvent;
			tempEvent.triangleIndex=triangleIndex;
			tempEvent.planeAxis=j;
			tempEvent.splitPlane=GetMinCoordinate[j];
			if(fabs(v1[j]-GetMinCoordinate[j])<PRECISION&&fabs(v2[j]-GetMinCoordinate[j])<PRECISION
															   &&fabs(v3[j]-GetMinCoordinate[j])<PRECISION)
			{
				tempEvent.triangleDirection=1;
				iniEvent->push_back(tempEvent);
				continue;
			}
			else if(fabs(v1[j]-GetMaxCoordinate[j])<PRECISION&&fabs(v2[j]-GetMaxCoordinate[j])<PRECISION
															   &&fabs(v3[j]-GetMaxCoordinate[j])<PRECISION)
			{
				tempEvent.splitPlane=GetMaxCoordinate[j];
				tempEvent.triangleDirection=1;
				iniEvent->push_back(tempEvent);
				continue;
			}
			else
			{
				tempEvent.triangleDirection=2;
				iniEvent->push_back(tempEvent);
				tempEvent.splitPlane=GetMaxCoordinate[j];
				tempEvent.triangleDirection=0;
				iniEvent->push_back(tempEvent);
			}
			
		}
}
void KdTree::UseSAHToCreateTree()
{
	//本程序默认0号结点为根结点
	memcpy(TreeNode[0].MinCoordinate,MinCoordinate,sizeof(TYPE)*DIMENS);
	memcpy(TreeNode[0].MaxCoordinate,MaxCoordinate,sizeof(TYPE)*DIMENS);
    vector<TriangleEvent> iniEvent;
    iniEvent.clear();
	for(int i=0;i<TreeNode[0].TriangleNumber;i++)
		SplitTriangle(TreeNode[0].trianglePtr[i],TreeNode[0].MinCoordinate,TreeNode[0].MaxCoordinate,&iniEvent);
	sort(iniEvent.begin(),iniEvent.end(),lessmark);
	SAHBuildTree(0,iniEvent,TreeNode[0].TriangleNumber*INTERSECTION_TIME,0);
 	this->NodeNumber=this->availableNode;
		ConvertIndexToLinear();
	 int temp[6]={-1,-1,-1,-1,-1,-1};
	 memcpy(TreeNode[0].RopeIndex,temp,sizeof(int)*6);
	 RopeConstruction(0,temp,MinCoordinate,MaxCoordinate);
}
void KdTree::transformTreeToGPU(){
	cudaMalloc((void **)&TreeNode_device,sizeof(Knode)*NodeNumber);		
	cudaError_t error =  cudaGetLastError();
	printf("kd-tree data transfer: %s\n",cudaGetErrorString(error));
	cudaMemcpy(TreeNode_device,TreeNode,sizeof(Knode)*NodeNumber,cudaMemcpyHostToDevice);


	cudaMalloc((void **)&treeTriangle_device,sizeof(Triangle) * HaveTriangle);
	cudaMemcpy(treeTriangle_device,treeTriangle,sizeof(Triangle) * HaveTriangle,cudaMemcpyHostToDevice);

	cudaMalloc((void **)&TriangleIndexUseForCuda_device,sizeof(int) * TriangleIndexCount);
	cudaMemcpy(TriangleIndexUseForCuda_device,TriangleIndexUseForCuda,sizeof(int) * TriangleIndexCount,cudaMemcpyHostToDevice);

}
void KdTree::SAHBuildTree(int index, vector<TriangleEvent> &pEvent,float FatherCost,int depth)
{
//	printf("%ld\n",index);
   TreeNode[index].leftChild=-1;
   TreeNode[index].rightChild=-1;
   if(TreeNode[index].TriangleNumber==0) return;
   int TempnPSide,leftTriangleNumber=0,rightTriangleNumber=0,planeTriangleNumber;
   float CurCost=this->SAHFindPlane(index,pEvent,TempnPSide,TreeNode[index].planeAxis,TreeNode[index].splitPlane,leftTriangleNumber,rightTriangleNumber,planeTriangleNumber);
   if(CurCost>=FatherCost||depth>=this->MaxDepth) return;
   leftTriangleNumber+=TempnPSide*planeTriangleNumber;
   rightTriangleNumber+=(1-TempnPSide)*planeTriangleNumber; 
   TreeNode[index].leftChild=this->GetAvailableNodeIndex();
   int leftChild=TreeNode[index].leftChild;
   TreeNode[leftChild].TriangleNumber=leftTriangleNumber;
   TreeNode[index].rightChild=this->GetAvailableNodeIndex();
   int rightChild=TreeNode[index].rightChild;
   TreeNode[rightChild].TriangleNumber=rightTriangleNumber;
   
   int tempLIndex=0,tempRIndex=0; 
   for(int i=0;i<TreeNode[index].TriangleNumber;i++)//将三角形按照所选定的Split平面分给2个子结点,预统计三角形为节点分配空间
   {
	   Triangle *p=&treeTriangle[TreeNode[index].trianglePtr[i]];
	   float v[3]={p->v1.x,p->v1.y,p->v1.z},v1[3]={p->v2.x,p->v2.y,p->v2.z},v2[3]={p->v3.x,p->v3.y,p->v3.z};
	   int Axis=TreeNode[index].planeAxis;
	   if(v[Axis]<TreeNode[index].splitPlane||v1[Axis]<TreeNode[index].splitPlane||v2[Axis]<TreeNode[index].splitPlane)
			tempLIndex++;
	   if(v[Axis]>TreeNode[index].splitPlane||v1[Axis]>TreeNode[index].splitPlane||v2[Axis]>TreeNode[index].splitPlane)
			tempRIndex++;
	   if(fabs(v[Axis]-TreeNode[index].splitPlane)<PRECISION&&fabs(v1[Axis]-TreeNode[index].splitPlane)<PRECISION
														    &&fabs(v2[Axis]-TreeNode[index].splitPlane)<PRECISION)
	   {
		   if(!TempnPSide)
			   tempLIndex++;
		   else 
			   tempRIndex++;
	   }
   }  
   TreeNode[leftChild].TriangleNumber=tempLIndex;
   TreeNode[leftChild].trianglePtr=new int[TreeNode[leftChild].TriangleNumber];
   TreeNode[rightChild].TriangleNumber=tempRIndex;
   TreeNode[rightChild].trianglePtr=new int[tempRIndex];
   tempLIndex=0,tempRIndex=0;
   for(int i=0;i<TreeNode[index].TriangleNumber;i++)//将三角形按照所选定的Split平面分给2个子结点
   {
	   Triangle *p=&treeTriangle[TreeNode[index].trianglePtr[i]];
	   float v[3]={p->v1.x,p->v1.y,p->v1.z},v1[3]={p->v2.x,p->v2.y,p->v2.z},v2[3]={p->v3.x,p->v3.y,p->v3.z};
	   int Axis=TreeNode[index].planeAxis;
	   if(v[Axis]<TreeNode[index].splitPlane||v1[Axis]<TreeNode[index].splitPlane||v2[Axis]<TreeNode[index].splitPlane)
		   TreeNode[leftChild].trianglePtr[tempLIndex++]=TreeNode[index].trianglePtr[i];
	   if(v[Axis]>TreeNode[index].splitPlane||v1[Axis]>TreeNode[index].splitPlane||v2[Axis]>TreeNode[index].splitPlane)
			TreeNode[rightChild].trianglePtr[tempRIndex++]=TreeNode[index].trianglePtr[i];
	   if(fabs(v[Axis]-TreeNode[index].splitPlane)<PRECISION&&fabs(v1[Axis]-TreeNode[index].splitPlane)<PRECISION
														    &&fabs(v2[Axis]-TreeNode[index].splitPlane)<PRECISION)
	   {
		   if(!TempnPSide)
			   TreeNode[leftChild].trianglePtr[tempLIndex++]=TreeNode[index].trianglePtr[i];
		   else 
			  TreeNode[rightChild].trianglePtr[tempRIndex++]=TreeNode[index].trianglePtr[i];
	   }
   }  
   ///ClassifyLeftRightBoth(T,E, ˆp) 将EVENT进行分类给左右子结点,从此往下都是为了计算子结点的event队列
   int eventSize=pEvent.size(); 
   vector<TriangleEvent>leftEvent;  leftEvent.clear();
   
   vector<TriangleEvent>tempLeftEvent;  tempLeftEvent.clear();
   vector<TriangleEvent>rightEvent; rightEvent.clear();
   vector<TriangleEvent>tempRightEvent; tempRightEvent.clear();
   vector<int>bothTriangle;	bothTriangle.clear();  
   for(int i=0;i<TreeNode[index].TriangleNumber;i++)//找出那些横跨左右子节点的三角形
   {
	   Triangle *p=&treeTriangle[TreeNode[index].trianglePtr[i]];
	   float v[3]={p->v1.x,p->v1.y,p->v1.z},v1[3]={p->v2.x,p->v2.y,p->v2.z},v2[3]={p->v3.x,p->v3.y,p->v3.z};
	   int Axis=TreeNode[index].planeAxis;
	   if(v[TreeNode[index].planeAxis]<=TreeNode[index].splitPlane&&v1[TreeNode[index].planeAxis]<=TreeNode[index].splitPlane&&v2[TreeNode[index].planeAxis]<=TreeNode[index].splitPlane)
			continue;
	   if(v[TreeNode[index].planeAxis]>=TreeNode[index].splitPlane&&v1[TreeNode[index].planeAxis]>=TreeNode[index].splitPlane&&v2[TreeNode[index].planeAxis]>=TreeNode[index].splitPlane)
		    continue;
	   bothTriangle.push_back(TreeNode[index].trianglePtr[i]);
   }
   for(int i=0;i<eventSize;i++)
   {
	   TriangleEvent tempEvent=(pEvent[i]);
	   Triangle *p=&treeTriangle[tempEvent.triangleIndex];
	   float v[3]={p->v1.x,p->v1.y,p->v1.z},v1[3]={p->v2.x,p->v2.y,p->v2.z},v2[3]={p->v3.x,p->v3.y,p->v3.z};
	   if(fabs(v[TreeNode[index].planeAxis]-TreeNode[index].splitPlane)<PRECISION&&fabs(v1[TreeNode[index].planeAxis]-TreeNode[index].splitPlane)<PRECISION
		       &&fabs(v2[TreeNode[index].planeAxis]-TreeNode[index].splitPlane)<PRECISION)
	   {
		   if(TempnPSide==0)
				 leftEvent.push_back(tempEvent);
		   else
			     rightEvent.push_back(tempEvent);
	   }
	   else  if(v[TreeNode[index].planeAxis]<=TreeNode[index].splitPlane&&v1[TreeNode[index].planeAxis]<=TreeNode[index].splitPlane&&v2[TreeNode[index].planeAxis]<=TreeNode[index].splitPlane)   
		    leftEvent.push_back(tempEvent);
	   else if(v[TreeNode[index].planeAxis]>=TreeNode[index].splitPlane&&v1[TreeNode[index].planeAxis]>=TreeNode[index].splitPlane&&v2[TreeNode[index].planeAxis]>=TreeNode[index].splitPlane)
		   rightEvent.push_back(tempEvent);
	   
   }
   //给左右子节点的上下界赋值
   memcpy(TreeNode[leftChild].MinCoordinate,TreeNode[index].MinCoordinate,sizeof(TYPE)*DIMENS);
   memcpy(TreeNode[leftChild].MaxCoordinate,TreeNode[index].MaxCoordinate,sizeof(TYPE)*DIMENS);
   TreeNode[leftChild].MaxCoordinate[TreeNode[index].planeAxis]=TreeNode[index].splitPlane;

   memcpy(TreeNode[rightChild].MinCoordinate,TreeNode[index].MinCoordinate,sizeof(TYPE)*DIMENS);
   memcpy(TreeNode[rightChild].MaxCoordinate,TreeNode[index].MaxCoordinate,sizeof(TYPE)*DIMENS);
   TreeNode[rightChild].MinCoordinate[TreeNode[index].planeAxis]=TreeNode[index].splitPlane;
   //对于跨俩侧的三角形，予以分割给左右子结点
   for(int i=0;i<bothTriangle.size();i++)
   {
	   SplitTriangle(bothTriangle[i],TreeNode[leftChild].MinCoordinate,TreeNode[leftChild].MaxCoordinate,&tempLeftEvent);
	   SplitTriangle(bothTriangle[i],TreeNode[rightChild].MinCoordinate,TreeNode[rightChild].MaxCoordinate,&tempRightEvent);
   }
   sort(tempLeftEvent.begin(),tempLeftEvent.end(),lessmark);
   sort(tempRightEvent.begin(),tempRightEvent.end(),lessmark);
   //分别对左右子结点进行归并排序
   vector<TriangleEvent>curLeftEvent; curLeftEvent.clear();
   unsigned  int iTemp=0,iLeft=0,iRight=0;
   //归并操作
   while(iTemp<tempLeftEvent.size()||iLeft<leftEvent.size())
   {
	   if(iTemp>=tempLeftEvent.size())
		   curLeftEvent.push_back(leftEvent[iLeft++]);
	   else if(iLeft>=leftEvent.size())
		   curLeftEvent.push_back(tempLeftEvent[iTemp++]);
	   else
	   {
		   if(tempLeftEvent[iTemp].planeAxis<leftEvent[iLeft].planeAxis)
				 curLeftEvent.push_back(tempLeftEvent[iTemp++]);
		   else if(tempLeftEvent[iTemp].planeAxis==leftEvent[iLeft].planeAxis&&tempLeftEvent[iTemp].splitPlane<leftEvent[iLeft].splitPlane)
			    curLeftEvent.push_back(tempLeftEvent[iTemp++]);
		   else if(tempLeftEvent[iTemp].planeAxis==leftEvent[iLeft].planeAxis&&fabs(tempLeftEvent[iTemp].splitPlane-leftEvent[iLeft].splitPlane)<PRECISION
			   &&tempLeftEvent[iTemp].triangleDirection<leftEvent[iLeft].triangleDirection)
               curLeftEvent.push_back(tempLeftEvent[iTemp++]);
		   else  curLeftEvent.push_back(leftEvent[iLeft++]); 
	   }
   }  
   iTemp=0;
   vector<TriangleEvent>curRightEvent; curRightEvent.clear();
   while(iTemp<tempRightEvent.size()||iRight<rightEvent.size())
   {
	   if(iTemp>=tempRightEvent.size())
		   curRightEvent.push_back(rightEvent[iRight++]);
	   else if(iRight>=rightEvent.size())
		   curRightEvent.push_back(tempRightEvent[iTemp++]);
	   else
	   {
		   if(tempRightEvent[iTemp].planeAxis<rightEvent[iRight].planeAxis)
				 curRightEvent.push_back(tempRightEvent[iTemp++]);
		   else if(tempRightEvent[iTemp].planeAxis==rightEvent[iRight].planeAxis&&tempRightEvent[iTemp].splitPlane<rightEvent[iRight].splitPlane)
			    curRightEvent.push_back(tempRightEvent[iTemp++]);
		   else if(tempRightEvent[iTemp].planeAxis==rightEvent[iRight].planeAxis&&fabs(tempRightEvent[iTemp].splitPlane-rightEvent[iRight].splitPlane)<PRECISION
			   &&tempRightEvent[iTemp].triangleDirection<rightEvent[iRight].triangleDirection)
               curRightEvent.push_back(tempRightEvent[iTemp++]);
		   else  curRightEvent.push_back(rightEvent[iRight++]); 
	   }
   }
 //  printf("index:%ld Triangle:%ld  father:%ld \nleft:%ld Triangle:%ld  right:%ld  Triangle:%ld \n",index,TreeNode[index].TriangleNumber,pEvent.size(),curLeftEvent.size(),TreeNode[leftChild].TriangleNumber,curRightEvent.size(),TreeNode[rightChild].TriangleNumber);
  
   vector<TriangleEvent>().swap(tempLeftEvent);
   vector<TriangleEvent>().swap(leftEvent);
   vector<TriangleEvent>().swap(tempRightEvent);
   vector<TriangleEvent>().swap(rightEvent);
   vector<TriangleEvent>().swap((pEvent));
   SAHBuildTree(leftChild,curLeftEvent,INTERSECTION_TIME*TreeNode[leftChild].TriangleNumber,depth+1);
   SAHBuildTree(rightChild,curRightEvent,INTERSECTION_TIME*TreeNode[rightChild].TriangleNumber,depth+1);
} 
float KdTree::SAHFindPlane(int nodeIndex,vector<TriangleEvent> &pEvent,int &nPSide, int &planeAxis, float &splitPlane, int &leftN, int &rightN, int &middleN)
{
	int Np[DIMENS]={0},Nl[DIMENS]={0},Nr[DIMENS]={TreeNode[nodeIndex].TriangleNumber,TreeNode[nodeIndex].TriangleNumber,TreeNode[nodeIndex].TriangleNumber};
	int pAdd,pPlane,pReduce;
	TriangleEvent compareEvent=pEvent[0];
	float BestCost=INF;
	//myDir.clear();
	for(unsigned int i=0;i<pEvent.size();)
	{
            pReduce=0;pPlane=0;pAdd=0;  
			while(i<pEvent.size()&&pEvent[i].planeAxis==compareEvent.planeAxis&&fabs(pEvent[i].splitPlane-compareEvent.splitPlane)<=1e-06
				&&pEvent[i].triangleDirection==0) 
				pReduce++,i++;
			while(i<pEvent.size()&&pEvent[i].planeAxis==compareEvent.planeAxis&&fabs(pEvent[i].splitPlane-compareEvent.splitPlane)<=1e-06
				&&pEvent[i].triangleDirection==1)
			{
			 //   if(pEvent[i].planeAxis==pEvent[151554].planeAxis)
			//	myDir[pEvent[i].triangleIndex]=1;
				pPlane++,i++;
			}
			while(i<pEvent.size()&&pEvent[i].planeAxis==compareEvent.planeAxis&&fabs(pEvent[i].splitPlane-compareEvent.splitPlane)<=1e-06
				&&pEvent[i].triangleDirection==2)
			{
			//	if(pEvent[i].planeAxis==pEvent[151554].planeAxis)
			//	myDir[pEvent[i].triangleIndex]=1;
			//	if(pEvent[i].planeAxis==1&&pEvent[i].triangleIndex==28002)
			//		int t=0;
				pAdd++,i++;
			}
			Np[compareEvent.planeAxis]=pPlane; Nr[compareEvent.planeAxis]-=pReduce;Nr[compareEvent.planeAxis]-=pPlane;
			int tempNpToSide;  	
		/*	int tempNp=0,tempNl=0,tempNr=0;
			if(i==151560)
			for(int j=0;j<TreeNode[nodeIndex].TriangleNumber;j++)
			 {
				 TriangleEvent temp=compareEvent;
				 compareEvent=pEvent[151554];
				 Triangle *p=&treeTriangle[TreeNode[nodeIndex].trianglePtr[j]];
				  float v[3]={p->v1.x,p->v1.y,p->v1.z},v1[3]={p->v2.x,p->v2.y,p->v2.z},v2[3]={p->v3.x,p->v3.y,p->v3.z}; 
				  if(fabs(v[compareEvent.planeAxis]-compareEvent.splitPlane)<PRECISION&&fabs(v1[compareEvent.planeAxis]-compareEvent.splitPlane)<PRECISION
					  &&fabs(v2[compareEvent.planeAxis]-compareEvent.splitPlane)<PRECISION) 
					   tempNp++;
				   if(v[compareEvent.planeAxis]<compareEvent.splitPlane||v1[compareEvent.planeAxis]<compareEvent.splitPlane
					   ||v2[compareEvent.planeAxis]<compareEvent.splitPlane)
				  {
					  if(!myDir[TreeNode[nodeIndex].trianglePtr[j]])
						  int tt=0;
					   tempNl++;
				  }
				  if(v[compareEvent.planeAxis]>compareEvent.splitPlane||v1[compareEvent.planeAxis]>compareEvent.splitPlane
					   ||v2[compareEvent.planeAxis]>compareEvent.splitPlane)
				       tempNr++;
                compareEvent=temp;
			 }

			if(i==151560)
			if(Np[compareEvent.planeAxis]!=tempNp||Nl[compareEvent.planeAxis]!=tempNl||Nr[compareEvent.planeAxis]!=tempNr)
				    printf("(%ld %ld %ld)  right:(%ld %ld %ld)\n",Nl[compareEvent.planeAxis],Np[compareEvent.planeAxis],Nr[compareEvent.planeAxis],tempNl,tempNp,tempNr);
			/**/
			float tempCost=this->SAHcost(tempNpToSide,compareEvent.planeAxis,compareEvent.splitPlane,
				                         TreeNode[nodeIndex].MinCoordinate,TreeNode[nodeIndex].MaxCoordinate,Nl[compareEvent.planeAxis],Nr[compareEvent.planeAxis],Np[compareEvent.planeAxis]);
			if(tempCost<BestCost&&fabs(compareEvent.splitPlane-TreeNode[nodeIndex].MinCoordinate[compareEvent.planeAxis])>PRECISION
								&&fabs(compareEvent.splitPlane-TreeNode[nodeIndex].MaxCoordinate[compareEvent.planeAxis])>PRECISION)
			{
				BestCost=tempCost;nPSide=tempNpToSide;
				planeAxis=compareEvent.planeAxis; splitPlane=compareEvent.splitPlane;
				leftN=Nl[planeAxis]; rightN=Nr[planeAxis];middleN=Np[planeAxis];
			}
			Nl[compareEvent.planeAxis]+=pAdd;Nl[compareEvent.planeAxis]+=pPlane;Np[compareEvent.planeAxis]=0;
			if(i<pEvent.size())
				compareEvent=pEvent[i];
	}
	return BestCost;
}
void KdTree::destoryTree()
{
	HaveNode=0;
	HaveTriangle=0;
	MaxStack=0;
	availableNode=1;
	delete []TriangleIndexUseForCuda;
	delete []TreeNode;
	delete []treeTriangle;
}