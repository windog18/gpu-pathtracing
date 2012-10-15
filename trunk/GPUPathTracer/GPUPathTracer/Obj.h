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


#define MAX_TEXTURES 100								// ����������Ŀ
#define BYTE unsigned char
// ����3D����࣬���ڱ���ģ���еĶ���
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

// ����2D���࣬���ڱ���ģ�͵�UV��������
class CVector2 
{
public:
	float x, y;
};

//  ��Ľṹ����
struct tFace
{
	int vertIndex[4];			// ��������
	int coordIndex[4];			// ������������
};

//  ������Ϣ�ṹ��
struct tMaterialInfo
{
	char  strName[255];			// ��������
	char  strFile[255];			// �����������ӳ�䣬���ʾ�����ļ�����
	//BYTE  color[3];				// �����RGB��ɫ
	float  Ks[3];               //������ɫ
	float  Kd[3];               //��������ɫ
	float  Ka[3];               //��������ɫ
	unsigned int   texureId;	// ����ID
	float uTile;				// u �ظ�
	float vTile;				// v �ظ�
	float uOffset;			    // u ����ƫ��
	float vOffset;				// v ����ƫ��
	float ns;
} ;

//  ������Ϣ�ṹ��
struct t3DObject 
{
	int  numOfVerts;			// ģ���ж������Ŀ
	int  numOfFaces;			// ģ���������Ŀ
	int  numTexVertex;			// ģ���������������Ŀ
	int  materialID;			// ����ID
	bool bHasTexture;			// �Ƿ��������ӳ��
	char strName[255];			// ���������
	CVector3  *pVerts;			// ����Ķ���
	CVector3  *pNormals;		// ����ķ�����
	CVector2  *pTexVerts;		// ����UV����
	tFace *pFaces;				// ���������Ϣ
};

//  ģ����Ϣ�ṹ��
struct t3DModel 
{
	int numOfObjects;					// ģ���ж������Ŀ
	int numOfMaterials;					// ģ���в��ʵ���Ŀ
	vector<tMaterialInfo> pMaterials;	// ����������Ϣ
	vector<t3DObject> pObject;			// ģ���ж���������Ϣ
	t3DModel(){
		numOfObjects = 0;
		numOfMaterials = 0;
		pMaterials.clear();
		pObject.clear();
	}
};

							


////maybe some bug because of using the ungetc()

// �����Ƕ���obj�ļ�����
class CLoadObj 
{
public:
	//  ��obj�ļ��е���Ϣ���뵽ģ����
	bool ImportObj(t3DModel *pModel,const char *strFileName);

	// ���������ImportObj()�н����øú���
	void ReadObjFile(t3DModel *pModel);

	// ���붥����Ϣ����ReadObjFile()�е��øú���
	void ReadVertexInfo();

	// ��������Ϣ����ReadObjFile()�е��øú���
	void ReadFaceInfo();

	// �������Ϣ�Ķ���֮����øú���
	void FillInObjectInfo(t3DModel *pModel);

	// ���㶥��ķ�����
	void ComputeNormals(t3DModel *pModel);

	// ������obj�ļ���û������/�������ƣ�ֻ���ֶ�����
	// materialID����ģ�Ͳ����б��е�������
	void SetObjectMaterial(t3DModel *pModel, int whichObject, int materialID);

	void LoadMaterialFile(t3DModel *pModel,const string materialFile);
private:
	map<string,int> materialNameToID;
	// ��Ҫ������ļ�ָ��
	FILE *m_FilePointer;

	// ��������
	vector<CVector3>  m_pVertices;

	// ������
	vector<tFace> m_pFaces;

	// UV��������
	vector<CVector2>  m_pTextureCoords;

	// ��ǰ�����Ƿ������������
	bool m_bObjectHasUV;

	// ��ǰ����Ķ����Ƿ�����
	bool m_bJustReadAFace;
	// ��ǰ������ж��㷨��
	bool m_bObjectHasN;
	//�쳣����
	char next;

	//��ǰ������ʱ��
	int materialID;
};
void DrawModel(t3DModel &pModel);

#endif
