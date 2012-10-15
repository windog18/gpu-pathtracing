#pragma once
#include "../../common/Vec3.h"
class Light
{
public:
	//���캯��
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

	Vec3 m_Pos;//��Դλ��
	Vec3 m_Color;//��Դ��ɫ
};