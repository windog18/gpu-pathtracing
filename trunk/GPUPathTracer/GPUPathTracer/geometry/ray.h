#pragma once
#include "Vec3.h"
#include <iostream>
class Ray
{
public:
	//构造函数
     __host__ __device__
	Ray()
	:segment(10000000),error_offset(0.001)
	 {}
     __host__ __device__
	Ray(const Vec3& _startPos, const Vec3& _unitdir)
	:	m_startPos(_startPos),
		m_unitDir(_unitdir),segment(10000000),error_offset(0.001)
	{}
     __host__ __device__
	Ray(const Ray& r)
	:	m_startPos(r.m_startPos),
		m_unitDir(r.m_unitDir),segment(r.segment),error_offset(r.error_offset)
	{}
     __host__ __device__
	~Ray(){}
     __host__ __device__
	Ray& operator=(const Ray& r){
		m_startPos=r.m_startPos;
		m_unitDir=r.m_unitDir;
		segment      = r.segment;
		error_offset = r.error_offset;
		return (*this);
	}

	__host__
	std::ostream& operator <<(std::ostream& os){
		os<<"startPosition:"<<m_startPos.x<<" "<<m_startPos.y<<" "<<m_startPos.z<<std::endl;
		os<<"dir:"<<m_unitDir.x<<" "<<m_unitDir.y<<" "<<m_unitDir.z<<std::endl;
		return os;
	}
	Vec3 m_startPos;//起点
	float error_offset;
	Vec3 m_unitDir;//单位方向向量
	float segment;
};