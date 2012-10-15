#pragma once
#include <float.h>
#include "ray.h"
class Triangle
{
public:
	Triangle(const Vec3 _v1,const Vec3 _v2,const Vec3 _v3,
		const Vec3 _n1,const Vec3 _n2,const Vec3 _n3,const int _textureIndex,const Vec3 _TexV1,const Vec3 _TexV2,const Vec3 _TexV3 )
		:v1(_v1),v2(_v2),v3(_v3),n1(_n1),n2(_n2),n3(_n3),textureIndex(_textureIndex),TexV1(_TexV1),TexV2(_TexV2),TexV3(_TexV3){
			index=-1;
			BL=Vec3(FLT_MAX,FLT_MAX,FLT_MAX);
			BH=Vec3(-FLT_MAX,-FLT_MAX,-FLT_MAX);
			calAABBox(v1);
			calAABBox(v2);
			calAABBox(v3);
			HasTexture=true;
	}
	Triangle(const Vec3 _v1,const Vec3 _v2,const Vec3 _v3,
		const Vec3 _n1,const Vec3 _n2,const Vec3 _n3)
		:v1(_v1),v2(_v2),v3(_v3),n1(_n1),n2(_n2),n3(_n3){
			index=-1;
			BL=Vec3(FLT_MAX,FLT_MAX,FLT_MAX);
			BH=Vec3(-FLT_MAX,-FLT_MAX,-FLT_MAX);
			calAABBox(v1);
			calAABBox(v2);
			calAABBox(v3);
			HasTexture=false;
	}
	Triangle(const Vec3 _v1,const Vec3 _v2,const Vec3 _v3)
		:v1(_v1),v2(_v2),v3(_v3){
			calNormal();
			index=-1;
			BL=Vec3(FLT_MAX,FLT_MAX,FLT_MAX);
			BH=Vec3(-FLT_MAX,-FLT_MAX,-FLT_MAX);
			calAABBox(v1);
			calAABBox(v2);
			calAABBox(v3);
			HasTexture=false;
	}
	Triangle(const Triangle& t){
		v1=t.v1;
		v2=t.v2;
		v3=t.v3;
		n1=t.n1;
		n2=t.n2;
		n3=t.n3;
		index=t.index;
		TexV1=t.TexV1;
		TexV2=t.TexV2;
		TexV3=t.TexV3;
		HasTexture=t.HasTexture;
		textureIndex=t.textureIndex;
		BL=Vec3(FLT_MAX,FLT_MAX,FLT_MAX);
		BH=Vec3(-FLT_MAX,-FLT_MAX,-FLT_MAX);
		calAABBox(v1);
		calAABBox(v2);
		calAABBox(v3);
	}
	Triangle():v1(Vec3()),v2(Vec3()),v3(Vec3()){}
	~Triangle(){}

	Triangle& operator = (const Triangle& t)
	{	
		v1=t.v1;
		v2=t.v2;
		v3=t.v3;
		n1=t.n1;
		n2=t.n2;
		n3=t.n3;
		index=t.index;
		TexV1=t.TexV1;
		TexV2=t.TexV2;
		TexV3=t.TexV3;
		HasTexture=t.HasTexture;
		textureIndex=t.textureIndex;
		BL=Vec3(FLT_MAX,FLT_MAX,FLT_MAX);
		BH=Vec3(-FLT_MAX,-FLT_MAX,-FLT_MAX);
		calAABBox(v1);
		calAABBox(v2);
		calAABBox(v3);
		return *this;
	}
    float getDistanceUntilHit(const Ray& ray)
	{
			Vec3 A=v2-v1,B=v3-v1,C=-1*ray.m_unitDir,D=ray.m_startPos-v1;
			float a=A.x,b=B.x,c=C.x,
			d=A.y,e=B.y,f=C.y,
			g=A.z,h=B.z,i=C.z,
			j=D.x,k=D.y,l=D.z;
			float det=a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g),
			det0=j*(e*i-f*h)-b*(k*i-f*l)+c*(k*h-e*l),
			det1=a*(k*i-f*l)-j*(d*i-f*g)+c*(d*l-k*g),
			det2=a*(e*l-k*h)-b*(d*l-k*g)+j*(d*h-e*g);
			if(fabs(det)<=1e-06) return -666.0f;
			float U=det0/det,V=det1/det,T=det2/det;
		    if(T>0.00001f&&-0.00001f<=U&&-0.00001f<=V&&U+V<=1.00001f)return T;
			//if(T>0&&0<=U&&U<=1&&0<=V&&V<=1&&0<=U+V&&U+V<=1)return T;
			else return -666.0f;/**/
	}
	Vec3 v1,v2,v3;
	Vec3 TexV1,TexV2,TexV3;//纹理坐标
	Vec3 n1,n2,n3;
	bool HasTexture;
	Vec3 BL,BH;//包围盒最小点、最大点
	int index;//三角形索引
	int textureIndex;
private:
	void calAABBox(Vec3 _v){
		BL.x=BL.x<_v.x ? BL.x:_v.x;
		BL.y=BL.y<_v.y ? BL.y:_v.y;
		BL.z=BL.z<_v.z ? BL.z:_v.z;
		BH.x=BH.x>_v.x ? BH.x:_v.x;
		BH.y=BH.y>_v.y ? BH.y:_v.y;
		BH.z=BH.z>_v.z ? BH.z:_v.z;
	}
	void calNormal(){
		n1=crossProduct(v3-v1,v2-v1);
		n2=crossProduct(v1-v2,v3-v2);
		n3=crossProduct(v2-v3,v1-v3);
		//n1=Vec3(1.0f,-1.0f,1.0f);
		//n2=Vec3(1.0f,1.0f,1.0f);
		//n3=Vec3(1.0f,1.0f,1.0f);
	}
};
//struct GTriangle{
//	Vec3 v1,v2,v3;
//	Vec3 n1,n2,n3;
//	Vec3 BL,BH;
//	Material mt;
//	int index;
//};
