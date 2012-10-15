#pragma once
#include "../../common/Vec3.h"
class Light
{
public:
	//构造函数
	Light():m_Pos(Vec3(0.0f,0.0f,0.0f)),m_Color(Vec3(1.0f,1.0f,1.0f)){}	
	Light(const Vec3& _lightPos, const Vec3& _Color)
	:	m_Pos(_lightPos),
		m_Color(_Color)
	{}

	~Light(){}

	Light& operator=(const Light& r){
		m_Pos=r.m_Pos;
		m_Color=r.m_Color;
		return (*this);
	}

	Vec3 m_Pos;//光源位置
	Vec3 m_Color;//光源颜色
};