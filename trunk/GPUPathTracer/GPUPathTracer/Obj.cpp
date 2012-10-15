
#include "Obj.h"
#include "sstream"


//  下面的函数的功能是将obj文件的信息读入指定的模型中
bool CLoadObj::ImportObj(t3DModel *pModel,const char *strFileName)
{
	char strMessage[255] = {0};				// 用于显示错误信息

	// 判断是否是一个合法的模型和文件类型
	if(!pModel || !strFileName) return false;

	// 以只读方式打开文件，返回文件指针
	m_FilePointer = fopen(strFileName, "rt+");

	// 判断文件指针是否正确
	if(!m_FilePointer) {
		// 如果文件指针不正确，则显示错误信息
		sprintf(strMessage, "Unable to find or open the file: %s", strFileName);
		MessageBox(NULL, strMessage, "Error", MB_OK);
		return false;
	}

	// 读入文件信息
	ReadObjFile(pModel);

	// 计算顶点的法向量，用于光照
	ComputeNormals(pModel);

	// 关闭打开的文件
	fclose(m_FilePointer);
		// 遍历所有的材质
	return true;
}


//  读入obj文件中的对象到模型中
void CLoadObj::ReadObjFile(t3DModel *pModel)
{
	char strLine[255]		= {0};
	char ch					= 0;
	next = 0;
	while(!feof(m_FilePointer))
	{
		float x = 0.0f, y = 0.0f, z = 0.0f;
		if(next == 0){
			// 获得obj文件中的当前行的第一个字符
			ch = fgetc(m_FilePointer);
		}else{
			ch = next;
		}
		next = 0;
		switch(ch)
		{
		case 'v':						// 读入的是'v' (后续的数据可能是顶点/法向量/纹理坐标)
			
			// 如果在前面读入的是面的行，那么现在进入的是另一个对象，因此在读入下一个对象之前，
			//  需要将最后对象的数据写入到模型结构体中
			if(m_bJustReadAFace) 
			{
				// 将最后对象的信息保存到模型结构中
				FillInObjectInfo(pModel);
			}

			// 读入点的信息，要区分顶点 ("v")、法向量 ("vn")、纹理坐标 ("vt")
			ReadVertexInfo();
			break;

		case 'f':						// 读入的是'f'
			
			//  读入面的信息
			ReadFaceInfo();
			break;

		default:
			//incase that it may be read to next line
			ungetc(ch,m_FilePointer);
			// 略过该行的内容
			fgets(strLine, 100, m_FilePointer);
			stringstream ss;
			ss<<strLine;
			string header;
			ss>>header;
			if(header == "mtllib"){
				string fileName;
				ss>>fileName;
				LoadMaterialFile(pModel,fileName);
			}else if(header == "usemtl"){
				string materialName;
				ss>>materialName;
				materialID = materialNameToID[materialName];
			}
			break;
		}
	}

	// 保存最后读入的对象
	FillInObjectInfo(pModel);
}


//  下面的函数读入顶点信息('v'是指顶点，'vt'指UV坐标)
void CLoadObj::ReadVertexInfo()
{
	CVector3 vNewVertex		= {0};
	CVector2 vNewTexCoord	= {0};
	char strLine[255]		= {0};
	char ch = 0;

	// 读入第一个字符，判断读入的是否顶点/法向量/UV坐标
	ch = fgetc(m_FilePointer);

	if(ch == ' ')				// 如果是空格，则必是顶点("v")
	{
		// 读入顶点坐标，格式是"v x y z"
		fscanf(m_FilePointer, "%f %f %f", &vNewVertex.x, &vNewVertex.y, &vNewVertex.z);

		// 读入该行中余下的内容，则文件指针到下一行
		fgets(strLine, 100, m_FilePointer);

		// 添加一个新的顶点到顶点链表中
		m_pVertices.push_back(vNewVertex);
	}
	else if(ch == 't')			// 如果是't'，则必定是纹理坐标("vt")
	{
		// 读入纹理坐标，格式是"vt u v"
		fscanf(m_FilePointer, "%f %f", &vNewTexCoord.x, &vNewTexCoord.y);

		// 读入该行余下的内容，则文件指针指向下一行
		fgets(strLine, 100, m_FilePointer);

		// 添加一个新的纹理坐标到链表中
		m_pTextureCoords.push_back(vNewTexCoord);

		//  设置对象具有纹理坐标为true
		m_bObjectHasUV = true;
	}
	else						// 否则可能是法向量("vn")
	{
		if(ch == 'n'){
			m_bObjectHasN = true;
		}
		if(ch == 't'){
			int l = 0;
		}
		::ungetc(ch,m_FilePointer);
		// 由于在最后计算各点的法向量，在这里略过
		fgets(strLine, 100, m_FilePointer);
	}
}


//  下面的函数读入面信息
void CLoadObj::ReadFaceInfo()
{
	tFace newFace			= {0};
	char strLine[255]		= {0};
	fscanf(m_FilePointer,"%s",strLine);
	newFace.vertIndex[3] = -1;
	if(m_pFaces.size() == 9204){
		int k =0;
	}
	int index = 0;
	do{
		std::stringstream ss;
		ss<<strLine;
		ss>>newFace.vertIndex[index];
		char temp;
		if(m_bObjectHasUV){
			ss>>temp>>newFace.coordIndex[index];
		}
		if(m_bObjectHasN){
			int tempNumber;
			ss>>temp>>tempNumber;
		}
		index++;

	if(fscanf(m_FilePointer,"%s",strLine)==EOF){
		m_pFaces.push_back(newFace);
		return ;
	}
	}while(strLine[0] >='0'&&strLine[0] <='9');
	if(index == 4 && m_pFaces.size()== 10000 - 1){
		int  k = 0;
	}
	if(m_FilePointer->_cnt == m_FilePointer->_bufsiz){
	   next = strLine[0];
	}else{
		ungetc(strLine[0],m_FilePointer);
	}
	// 读入该行余下的内容，则文件指针指向下一行
	//fgets(strLine, 100, m_FilePointer);
				
	// 添加一个新面到面链表中
	m_pFaces.push_back(newFace);

	//  设置刚才读入的是面
	m_bJustReadAFace = true;
}


//  下面的函数将读入对象的信息写入模型结构体中
void CLoadObj::FillInObjectInfo(t3DModel *pModel)
{
	t3DObject newObject = {0};
	int textureOffset = 0, vertexOffset = 0;
	int i = 0;

	// 模型中对象计数器递增
	pModel->numOfObjects++;

	// 添加一个新对象到模型的对象链表中
	pModel->pObject.push_back(newObject);

	// 获得当前对象的指针
	t3DObject *pObject = &(pModel->pObject[pModel->numOfObjects - 1]);

	// 获得面的数量、顶点的数量和纹理坐标的数量
	pObject->numOfFaces   = m_pFaces.size();
	pObject->numOfVerts   = m_pVertices.size();
	pObject->numTexVertex = m_pTextureCoords.size();

	// 如果读入了面
	if(pObject->numOfFaces) 
	{

		// 分配保存面的存储空间
		pObject->pFaces = new tFace [pObject->numOfFaces];
	}

	// 如果读入了点
	if(pObject->numOfVerts) {

		// 分配保存点的存储空间
		pObject->pVerts = new CVector3 [pObject->numOfVerts];
	}	

	// 如果读入了纹理坐标
	if(pObject->numTexVertex) {
		pObject->pTexVerts = new CVector2 [pObject->numTexVertex];
		pObject->bHasTexture = true;
	}	

	// 遍历所有的面
	for(i = 0; i < pObject->numOfFaces; i++)
	{
		// 拷贝临时的面链表到模型链表中
		pObject->pFaces[i] = m_pFaces[i];

		// 判断是否是对象的第一个面
		if(i == 0) 
		{
			int k = 0;
			vertexOffset = pObject->pFaces[0].vertIndex[0];
			for(int tIdx = 1;tIdx <4;tIdx++) if(pObject->pFaces[0].vertIndex[tIdx]>=0){
				if(pObject->pFaces[0].vertIndex[tIdx] < vertexOffset){
					vertexOffset = pObject->pFaces[0].vertIndex[tIdx];
					k = tIdx;
				}
			}
			// 对于纹理坐标，也进行同样的操作
			if(pObject->numTexVertex > 0) {
				// 当前的索引剪去1
				textureOffset = pObject->pFaces[0].coordIndex[k];
			}					
		}

		for(int j = 0; j < 4; j++)
		{
			//  对于每一个索引，必须将其减去1
			pObject->pFaces[i].vertIndex[j]  -= vertexOffset;
			pObject->pFaces[i].coordIndex[j] -= textureOffset;
		}
	}

	// 遍历对象中的所有点
	for(i = 0; i < pObject->numOfVerts; i++)
	{
		// 将当前的顶点从临时链表中拷贝到模型链表中
		pObject->pVerts[i] = m_pVertices[i];
	}

	// 遍历对象中所有的纹理坐标
	for(i = 0; i < pObject->numTexVertex; i++)
	{
		// 将当前的纹理坐标从临时链表中拷贝到模型链表中
		pObject->pTexVerts[i] = m_pTextureCoords[i];
	}

	//  由于OBJ文件中没有材质，因此将materialID设置为-1，必须手动设置材质
	pObject->materialID = materialID - 1;

	//  清除所有的临时链表
	m_pVertices.clear();
	m_pFaces.clear();
	m_pTextureCoords.clear();

	// 设置所有的布尔值为false
	m_bObjectHasUV   = false;
	m_bJustReadAFace = false;
	m_bObjectHasN    = false;
}


//  下面的函数为对象序列中的对象赋予具体的材质
void CLoadObj::SetObjectMaterial(t3DModel *pModel, int whichObject, int materialID)
{
	// 确保模型合法
	if(!pModel) return;

	// 确保对象合法
	if(whichObject >= pModel->numOfObjects) return;

	// 给对象赋予材质ID
	pModel->pObject[whichObject].materialID = materialID;
}



void CLoadObj::LoadMaterialFile(t3DModel *pModel,const string materialFile){
	ifstream fin;
	fin.open(materialFile);
	this->materialNameToID.clear();
		if(!fin) return ;
	string header;
	int index = 1;
	while(fin>>header){
		if(header=="newmtl"){
			tMaterialInfo newMaterial;
			string tempBuffer;
			fin>>newMaterial.strName;

			while(fin>>tempBuffer&&tempBuffer!="endmtl"){
				if(tempBuffer=="Ka"){
					fin>>newMaterial.Ka[0]>>newMaterial.Ka[1]>>newMaterial.Ka[2];
				}
				else if(tempBuffer=="Kd")
					fin>>newMaterial.Kd[0]>>newMaterial.Kd[1]>>newMaterial.Kd[2];
				else if(tempBuffer == "Ks"){
					fin>>newMaterial.Ks[0]>>newMaterial.Ks[1]>>newMaterial.Ks[2];
				}else if(tempBuffer == "map_Kd"){
					fin>>tempBuffer;
					strcpy(newMaterial.strFile ,tempBuffer.c_str());
					CreateTexture(newMaterial.strFile,newMaterial.texureId);
				}else if(tempBuffer == "newmtl"){
					for(int i = tempBuffer.size() - 1 ;i >= 0;i--){
						fin.putback(tempBuffer[i]);
					}
					break;
				}
				
			}
			pModel->pMaterials.push_back(newMaterial);
			materialNameToID[string(newMaterial.strName)] = index++;
	/*		cout<<"header: "<<newMaterial.materialName<<endl;
			cout<<newMaterial.materialType<<endl;
			cout<<newMaterial.DR.x<<" "<<newMaterial.DR.y<<" "<<newMaterial.DR.z<<endl;*/

		}
	}
	pModel->numOfMaterials = index - 1;
}

//  下面的这些函数主要用来计算顶点的法向量，顶点的法向量主要用来计算光照
// 下面的宏定义计算一个矢量的长度
#define Mag(Normal) (sqrt(Normal.x*Normal.x + Normal.y*Normal.y + Normal.z*Normal.z))

// 下面的函数求两点决定的矢量
CVector3 Vector(CVector3 vPoint1, CVector3 vPoint2)
{
	CVector3 vVector;							

	vVector.x = vPoint1.x - vPoint2.x;			
	vVector.y = vPoint1.y - vPoint2.y;			
	vVector.z = vPoint1.z - vPoint2.z;			

	return vVector;								
}


// 下面的函数两个矢量相加
CVector3 AddVector(CVector3 vVector1, CVector3 vVector2)
{
	CVector3 vResult;							
	
	vResult.x = vVector2.x + vVector1.x;		
	vResult.y = vVector2.y + vVector1.y;		
	vResult.z = vVector2.z + vVector1.z;		

	return vResult;								
}

// 下面的函数处理矢量的缩放
CVector3 DivideVectorByScaler(CVector3 vVector1, float Scaler)
{
	CVector3 vResult;							
	
	vResult.x = vVector1.x / Scaler;			
	vResult.y = vVector1.y / Scaler;			
	vResult.z = vVector1.z / Scaler;			

	return vResult;								
}

// 下面的函数返回两个矢量的叉积
CVector3 Cross(CVector3 vVector1, CVector3 vVector2)
{
	CVector3 vCross;								
												
	vCross.x = ((vVector1.y * vVector2.z) - (vVector1.z * vVector2.y));
												
	vCross.y = ((vVector1.z * vVector2.x) - (vVector1.x * vVector2.z));
												
	vCross.z = ((vVector1.x * vVector2.y) - (vVector1.y * vVector2.x));

	return vCross;								
}


// 下面的函数规范化矢量
CVector3 Normalize(CVector3 vNormal)
{
	double Magnitude;							

	Magnitude = Mag(vNormal);					// 获得矢量的长度

	vNormal.x /= (float)Magnitude;				
	vNormal.y /= (float)Magnitude;				
	vNormal.z /= (float)Magnitude;				

	return vNormal;								
}

//  下面的函数用于计算对象的法向量
void CLoadObj::ComputeNormals(t3DModel *pModel)
{
	int i;
	CVector3 vVector1, vVector2, vNormal, vPoly[3];

	// 如果模型中没有对象，则返回
	if(pModel->numOfObjects <= 0)
		return;

	// 遍历模型中所有的对象
	for(int index = 0; index < pModel->numOfObjects; index++)
	{
		// 获得当前的对象
		t3DObject *pObject = &(pModel->pObject[index]);

		// 分配需要的存储空间
		CVector3 *pNormals		= new CVector3 [pObject->numOfFaces];
		CVector3 *pTempNormals	= new CVector3 [pObject->numOfFaces];
		pObject->pNormals		= new CVector3 [pObject->numOfVerts];

		// 遍历对象的所有面
		for(int i=0; i < pObject->numOfFaces; i++)
		{												
			vPoly[0] = pObject->pVerts[pObject->pFaces[i].vertIndex[0]];
			vPoly[1] = pObject->pVerts[pObject->pFaces[i].vertIndex[1]];
			vPoly[2] = pObject->pVerts[pObject->pFaces[i].vertIndex[2]];

			// 计算面的法向量

			vVector1 = Vector(vPoly[0], vPoly[2]);		// 获得多边形的矢量
			vVector2 = Vector(vPoly[2], vPoly[1]);		// 获得多边形的第二个矢量

			vNormal  = Cross(vVector1, vVector2);		// 获得两个矢量的叉积
			pTempNormals[i] = vNormal;					// 保存非规范化法向量
			vNormal  = Normalize(vNormal);				// 规范化获得的叉积

			pNormals[i] = vNormal;						// 将法向量添加到法向量列表中
		}

		//  下面求顶点法向量
		CVector3 vSum = {0.0, 0.0, 0.0};
		CVector3 vZero = vSum;
		int shared=0;
		// 遍历所有的顶点
		for (i = 0; i < pObject->numOfVerts; i++)			
		{
			for (int j = 0; j < pObject->numOfFaces; j++)	// 遍历所有的三角形面
			{												// 判断该点是否与其它的面共享
				if (pObject->pFaces[j].vertIndex[0] == i || 
					pObject->pFaces[j].vertIndex[1] == i || 
					pObject->pFaces[j].vertIndex[2] == i)
				{
					vSum = AddVector(vSum, pTempNormals[j]);
					shared++;								
				}
			}      
			
			pObject->pNormals[i] = DivideVectorByScaler(vSum, float(-shared));

			// 规范化最后的顶点法向
			pObject->pNormals[i] = Normalize(pObject->pNormals[i]);	

			vSum = vZero;								
			shared = 0;										
		}
	
		// 释放存储空间，开始下一个对象
		delete [] pTempNormals;
		delete [] pNormals;
	}
}
void DrawModel(t3DModel &g_3DModel){
	// 遍历模型中所有的对象
	for(int i = 1; i < g_3DModel.numOfObjects; i++)
	{
		// 如果对象的大小小于0，则退出
		if(g_3DModel.pObject.size() <= 0) break;

		// 获得当前显示的对象
		t3DObject *pObject = &g_3DModel.pObject[i];
			
		// 开始以g_ViewMode模式绘制
		if(pObject->pFaces[0].vertIndex[3]>=0)
			glBegin(GL_QUADS);
		else
			glBegin(GL_TRIANGLES);
			// 遍历所有的面
			for(int j = 0; j < pObject->numOfFaces; j++)
			{
				// 遍历三角形的所有点
				for(int whichVertex = 0; whichVertex < 4; whichVertex++)
				{
					// 获得面对每个点的索引
					int index = pObject->pFaces[j].vertIndex[whichVertex];
					if(index < 0) continue;
					// 给出法向量
					glNormal3f(pObject->pNormals[ index ].x, pObject->pNormals[ index ].y, pObject->pNormals[ index ].z);
				
					// 如果对象具有纹理
					if(pObject->bHasTexture) {

						// 确定是否有UVW纹理坐标
						if(pObject->pTexVerts) {
							glTexCoord2f(pObject->pTexVerts[ index ].x, pObject->pTexVerts[ index ].y);
						}
					} else {

						if(g_3DModel.pMaterials.size() && pObject->materialID >= 0) 
						{
							glColor3f(g_3DModel.pMaterials[pObject->materialID].Kd[0],
								      g_3DModel.pMaterials[pObject->materialID].Kd[1], 
									  g_3DModel.pMaterials[pObject->materialID].Kd[2]);
						}
					}
					glVertex3f(pObject->pVerts[ index ].x, pObject->pVerts[ index ].y, pObject->pVerts[ index ].z);
				}
			}

		glEnd();								// 绘制结束
	}

}

