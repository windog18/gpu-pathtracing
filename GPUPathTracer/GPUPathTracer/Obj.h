#ifndef _OBJ_H
#define _OBJ_H

#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <vector>
#include <gl\gl.h>										
#include <gl\glu.h>										
#include <crtdbg.h>
#include "math_ops.h"
#include <map>
using namespace std;

#define SCREEN_WIDTH 400								
#define SCREEN_HEIGHT 300								
#define SCREEN_DEPTH 16									


#define MAX_TEXTURES 100								// 最大的纹理数目
#define BYTE unsigned char
// 定义3D点的类，用于保存模型中的顶点
class CVector3 
{
public:
	float x, y, z;
	CVector3 operator +(const CVector3& op) const{
		CVector3 result;
		result.x = x + op.x;
		result.y = y + op.y;
		result.z = z + op.z;
		return result;
	}
	CVector3 operator /(float div) const{
		CVector3 result;
		result.x = x / div;
		result.y = y / div;
		result.z = z / div;
		return result;
	}
	CVector3 operator *(float mut) const{
		CVector3 result;
		result.x = x * mut;
		result.y = y * mut;
		result.z = z * mut;
		return result;
	}
	CVector3 operator -(const CVector3& op)const{
		CVector3 result;
		result.x = x - op.x;
		result.y = y - op.y;
		result.z = z - op.z;
		return result;
	}
	static CVector3 Zeros(){
		CVector3 result;
		result.x = 0;
		result.y = 0;
		result.z = 0;
		return result;
	}
	CVector3 &operator +=(const CVector3& op){
		x +=op.x;
		y +=op.y;
		z +=op.z;
		return (*this);
	}
	CVector3 & operator /=(float div){
		x /= div;
		y /= div;
		z /= div;
		return (*this);
	}
	float Length() const {
		return sqrtf(x*x + y*y + z*z);
	}
};

// 定义2D点类，用于保存模型的UV纹理坐标
class CVector2 
{
public:
	float x, y;
};

//  面的结构定义
struct tFace
{
	int vertIndex[4];			// 顶点索引
	int coordIndex[4];			// 纹理坐标索引
};

//  材质信息结构体
struct tMaterialInfo
{
	char  strName[255];			// 纹理名称
	char  strFile[255];			// 如果存在纹理映射，则表示纹理文件名称
	//BYTE  color[3];				// 对象的RGB颜色
	float  Ks[3];               //镜面颜色
	float  Kd[3];               //漫反射颜色
	float  Ka[3];               //环境光颜色
	unsigned int   texureId;	// 纹理ID
	float uTile;				// u 重复
	float vTile;				// v 重复
	float uOffset;			    // u 纹理偏移
	float vOffset;				// v 纹理偏移
	float ns;
} ;

//  对象信息结构体
struct t3DObject 
{
	int  numOfVerts;			// 模型中顶点的数目
	int  numOfFaces;			// 模型中面的数目
	int  numTexVertex;			// 模型中纹理坐标的数目
	int  materialID;			// 纹理ID
	bool bHasTexture;			// 是否具有纹理映射
	char strName[255];			// 对象的名称
	CVector3  *pVerts;			// 对象的顶点
	CVector3  *pNormals;		// 对象的法向量
	CVector2  *pTexVerts;		// 纹理UV坐标
	tFace *pFaces;				// 对象的面信息
};

//  模型信息结构体
struct t3DModel 
{
	int numOfObjects;					// 模型中对象的数目
	int numOfMaterials;					// 模型中材质的数目
	vector<tMaterialInfo> pMaterials;	// 材质链表信息
	vector<t3DObject> pObject;			// 模型中对象链表信息
	t3DModel(){
		numOfObjects = 0;
		numOfMaterials = 0;
		pMaterials.clear();
		pObject.clear();
	}
};

							


////maybe some bug because of using the ungetc()

// 下面是读入obj文件的类
class CLoadObj 
{
public:
	//  将obj文件中的信息读入到模型中
	bool ImportObj(t3DModel *pModel,const char *strFileName);

	// 读入对象，在ImportObj()中将调用该函数
	void ReadObjFile(t3DModel *pModel);

	// 读入顶点信息，在ReadObjFile()中调用该函数
	void ReadVertexInfo();

	// 读入面信息，在ReadObjFile()中调用该函数
	void ReadFaceInfo();

	// 完成面信息的读入之后调用该函数
	void FillInObjectInfo(t3DModel *pModel);

	// 计算顶点的法向量
	void ComputeNormals(t3DModel *pModel);

	// 由于在obj文件中没有纹理/材质名称，只能手动设置
	// materialID是在模型材质列表中的索引号
	void SetObjectMaterial(t3DModel *pModel, int whichObject, int materialID);

	void LoadMaterialFile(t3DModel *pModel,const string materialFile);
private:
	map<string,int> materialNameToID;
	// 需要读入的文件指针
	FILE *m_FilePointer;

	// 顶点链表
	vector<CVector3>  m_pVertices;

	// 面链表
	vector<tFace> m_pFaces;

	// UV坐标链表
	vector<CVector2>  m_pTextureCoords;

	// 当前对象是否具有纹理坐标
	bool m_bObjectHasUV;

	// 当前读入的对象是否是面
	bool m_bJustReadAFace;
	// 当前对象具有顶点法线
	bool m_bObjectHasN;
	//异常处理
	char next;

	//当前对象材质编号
	int materialID;
};
void DrawModel(t3DModel &pModel);

#endif
