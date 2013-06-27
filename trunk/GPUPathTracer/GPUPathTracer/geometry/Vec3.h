#pragma once
#include"assert.h"
#include <string>
#include "cuda_runtime.h"
#define Colour Vec3
/*=====================================================================
三维向量
可表示点或方向等
=====================================================================*/
 __host__ __device__   
__inline float RSqrt( float number ) 
{
	long i;
	float x2, y;
	const float threehalfs = 1.5f;

	x2 = number * 0.5f;
	y  = number;
	i  = * (long *) &y;			// evil floating point bit level hacking
	i  = 0x5f3759df - (i >> 1);             // what the f..k?
	y  = * (float *) &i;
	y  = y * (threehalfs - (x2 * y * y));   // 1st iteration

	return y;
}
class Vec3
{
public:
     __host__ __device__
	inline Vec3()
	{}
     __host__ __device__
	inline ~Vec3()
	{}
     __host__ __device__
	inline Vec3(float x_, float y_, float z_)
	:	x(x_),
		y(y_),
		z(z_)
	{}
     __host__ __device__   
	inline Vec3(const Vec3& rhs)
	:	x(rhs.x),
		y(rhs.y),
		z(rhs.z)
	{}
     __host__ __device__
	inline Vec3(const float* f)
	:	x(f[0]),
		y(f[1]),
		z(f[2])
	{}
     __host__ __device__
	inline Vec3(const Vec3& v, float scale)
	:	x(v.x * scale),
		y(v.y * scale),
		z(v.z * scale)
	{}
		
     __host__ __device__
	inline void set(float newx, float newy, float newz)
	{
		x = newx;
		y = newy;
		z = newz;
	}
     __host__ __device__
	inline float& operator[] (int index)
	{
		//NOTE: no asserting
		return ((float*)(&x))[index];
	}
     __host__ __device__
	inline const float& operator[] (int index) const
	{
		//NOTE: no asserting
		return ((float*)(&x))[index];
	}

__device__ __host__ 	inline const Vec3 operator + (const Vec3& rhs) const
	{
		return Vec3(x + rhs.x, y + rhs.y, z + rhs.z);
	}


__device__ __host__ 	
    inline const Vec3 operator - (const Vec3& rhs) const
	{	
		return Vec3(x - rhs.x, y - rhs.y, z - rhs.z);
	}

    __device__ __host__ 	
    inline const Vec3 operator * (const Vec3& rhs) const
	{	
		return Vec3(x * rhs.x, y * rhs.y, z * rhs.z);
	}



__device__ __host__ 	
	inline Vec3& operator += (const Vec3& rhs)
	{		
		x += rhs.x;
		y += rhs.y;
		z += rhs.z;
		return *this;
	}
     __host__ __device__
	inline Vec3& operator -= (const Vec3& rhs)
	{	
		x -= rhs.x;
		y -= rhs.y;
		z -= rhs.z;
		return *this;
	}

__device__ __host__ 	
	inline Vec3& operator = (const Vec3& rhs)
	{	
		x = rhs.x;
		y = rhs.y;
		z = rhs.z;
		return *this;
	}
__device__ __host__ 	
	inline Vec3& operator = (const float3& rhs)
	{	
		x = rhs.x;
		y = rhs.y;
		z = rhs.z;
		return *this;
	}
     __host__ __device__
	inline bool operator == (const Vec3& rhs) const
	{
		return ( (x == rhs.x) && (y == rhs.y) && (z == rhs.z) );
	}
     __host__ __device__
	inline bool operator != (const Vec3& rhs) const
	{
		return ( (x != rhs.x) || (y != rhs.y) || (z != rhs.z) );
	}
     __host__ __device__
	//for sorting Vec3's
	inline bool operator < (const Vec3& rhs) const
	{
		if(x < rhs.x)
			return true;
		else if(x > rhs.x)
			return false;
		else	//else x == rhs.x
		{
			if(y < rhs.y)
				return true;
			else if(y > rhs.y)
				return false;
			else
			{
				/*if(z < rhs.z)
					return true;
				else if(z >= rhs.z)
					return false;*/
				return z < rhs.z;
			}
		}
	}

__device__ __host__  inline Vec3 normalise()
	{
		//if(!x && !y && !z)
		//	return;

		float inverselength = length();//will be inverted later

		if(!inverselength)
			return Vec3(0.0f,0.0f,0.0f);

		inverselength = 1.0f / inverselength;//invert it

		x *= inverselength;
		y *= inverselength;
		z *= inverselength;

		return *this;
	}

     __host__ __device__   
	inline void fastNormalise()
	{
		const float inverselength = RSqrt( length2() );

		//if(!inverselength)
		//	return;

		x *= inverselength;
		y *= inverselength;
		z *= inverselength;
	}
     __host__ __device__   
	inline float normalise_ret_length()
	{
		//if(!x && !y && !z)
		//	return 0.0f;

		const float len = length();

		if(!len)
			return 0.00001f;

		const float inverselength = 1.0f / len;

		x *= inverselength;
		y *= inverselength;
		z *= inverselength;

		return len;
	}
     __host__ __device__   
	inline float normalise_ret_length(float& inv_len_out)
	{
		//if(!x && !y && !z)
		//	return 0.0f;

		const float len = length();

		if(!len)
			return 0.00001f;

		const float inverselength = 1.0f / len;

		x *= inverselength;
		y *= inverselength;
		z *= inverselength;

		inv_len_out = inverselength;

		return len;
	}
     __host__ __device__   
	inline float normalise_ret_length2()
	{
		//if(!x && !y && !z)
		//	return 0.0f;

		const float len2 = length2();

		if(!len2)
			return 0.00001f;

		const float inverselength = 1.0f / sqrt(len2);

		x *= inverselength;
		y *= inverselength;
		z *= inverselength;

		return len2;
	}




	__device__ __host__		
	inline float length()  const
	{
		return sqrt(x*x + y*y + z*z);
	}

	__device__ __host__		
	inline float length2() const
	{
		return (x*x + y*y + z*z);
	}
     __host__ __device__
	inline void scale(float factor)
	{
		x *= factor;
		y *= factor;
		z *= factor;
	}
     __host__ __device__
	inline Vec3& operator *= (float factor)
	{
		x *= factor;
		y *= factor;
		z *= factor;
		return *this;
	}
     __host__ __device__
	inline void setLength(float newlength)
	{
		const float current_len = length();

		if(!current_len)
			return;

		scale(newlength / current_len);
	}
     __host__ __device__
	inline Vec3& operator /= (float divisor)
	{
		*this *= (1.0f / divisor);
		return *this;
	}
     __host__ __device__
	inline const Vec3 operator * (float factor) const
	{
		return Vec3(x * factor, y * factor, z * factor);
	}
     __host__ __device__
	inline const Vec3 operator / (float divisor) const
	{
		const float inverse_d = (1.0f / divisor);

		return Vec3(x * inverse_d, y * inverse_d, z * inverse_d);
	}
     __host__ __device__
	inline void zero()
	{
		x = 0.0f;
		y = 0.0f;
		z = 0.0f;
	}
     __host__ __device__
	inline float getDist(const Vec3& other) const
	{
		const Vec3 dif = other - *this;
		return dif.length();
	}
     __host__ __device__
	inline float getDist2(const Vec3& other) const
	{
		//const Vec3 dif = other - *this;
		//return dif.length2();
		//float sum = other.x - x;

		//sum += other.y - y;

		//sum += other.z - z;

		float sum = other.x - x;
		sum *= sum;

		float dif = other.y - y;
		sum += dif*dif;

		dif = other.z - z;

		return sum + dif*dif;


		//return (other.x - x) + (other.y - y) + (other.z - z);
	}
     __host__ __device__
	inline void assertUnitVector() const
	{
		const float len = length();

		const float var = fabs(1.0f - len);

		const float EPSILON_ = 0.0001f;

		assert(var <= EPSILON_);
	}

	void print() const;
	const std::string toString() const;
	__host__ __device__ 
	inline float3 toFloat3(){
		return make_float3(x,y,z);
	}
//	const static Vec3 zerovector;	//(0,0,0)
//	const static Vec3 i;			//(1,0,0)
//	const static Vec3 j;			//(0,1,0)
//	const static Vec3 k;			//(0,0,1)

	float x,y,z;



	inline const float* data() const { return (float*)this; }
	



	//-----------------------------------------------------------------
	//Euler angle stuff
	//-----------------------------------------------------------------
	//static Vec3 ws_up;//default z axis
	//static Vec3 ws_right;//default -y axis
	//static Vec3 ws_forwards;//must = crossProduct(ws_up, ws_right).   default x axis

	//static void setWsUp(const Vec3& vec){ ws_up = vec; }
	//static void setWsRight(const Vec3& vec){ ws_right = vec; }
	//static void setWsForwards(const Vec3& vec){ ws_forwards = vec; }

	float getYaw() const { return x; }
	float getPitch() const { return y; }
	float getRoll() const { return z; }

	void setYaw(float newyaw){ x = newyaw; }
	void setPitch(float newpitch){ y = newpitch; }
	void setRoll(float newroll){ z = newroll; }

	/*==================================================================
	getAngles
	---------
	Gets the Euler angles of this vector.  Returns the vector (yaw, pitch, roll). 
	Yaw is the angle between this vector and the vector 'ws_forwards' measured
	anticlockwise when looking towards the origin from along the vector 'ws_up'.
	Yaw will be in the range (-Pi, Pi).

	Pitch is the angle between this vector and the vector 'ws_forwards' as seen when
	looking from along the vecotr 'ws_right' towards the origin.  A pitch of Pi means
	the vector is pointing along the vector 'ws_up', a pitch of -Pi means the vector is
	pointing straight down. (ie pointing in the opposite direction from 'ws_up'.
	A pitch of 0 means the vector is in the 'ws_right'-'ws_forwards' plane.
	Will be in the range [-Pi/2, Pi/2].

	Roll will be 0.
	====================================================================*/
	const Vec3 getAngles(const Vec3& ws_forwards, const Vec3& ws_up, const Vec3& ws_right) const;
	//const Vec3 getAngles() const; //around i, j, k

	const Vec3 fromAngles(const Vec3& ws_forwards, const Vec3& ws_up, const Vec3& ws_right) const;

__device__  __host__	float dotProduct(const Vec3& rhs) const
	{
		return x*rhs.x + y*rhs.y + z*rhs.z;
	}

__device__  __host__	float dot(const Vec3& rhs) const
	{
		return dotProduct(rhs);
	}

	static const Vec3 randomVec(float component_lowbound, float component_highbound);


__device__  __host__	inline void setToMult(const Vec3& other, float factor)
	{
		x = other.x * factor;
		y = other.y * factor;
		z = other.z * factor;
	}

	inline void addMult(const Vec3& other, float factor)
	{
		x += other.x * factor;
		y += other.y * factor;
		z += other.z * factor;
	}

	inline void subMult(const Vec3& other, float factor)
	{
		x -= other.x * factor;
		y -= other.y * factor;
		z -= other.z * factor;
	}

	inline void add(const Vec3& other)
	{
		x += other.x;
		y += other.y;
		z += other.z;
	}

	inline void sub(const Vec3& other)
	{
		x -= other.x;
		y -= other.y;
		z -= other.z;
	}

	inline void removeComponentInDir(const Vec3& unitdir)
	{
		subMult(unitdir, this->dot(unitdir));
	}
};

 __device__ __host__	 inline const Vec3 normalise(const Vec3& v)
{
	const float vlen = v.length();

	if(!vlen)
		return Vec3(1.0f, 0.0f, 0.0f);

	return v * (1.0f / vlen);
}

inline const Vec3 operator * (float m, const Vec3& right)
{
	return Vec3(right.x * m, right.y * m, right.z * m);
}


__device__ __host__ inline float dotProduct(const Vec3& v1, const Vec3& v2)
{
	return (v1.x * v2.x) + (v1.y * v2.y) + (v1.z * v2.z);
}

__device__ __host__ inline float dot(const Vec3& v1, const Vec3& v2)
{
	return (v1.x * v2.x) + (v1.y * v2.y) + (v1.z * v2.z);
}


__device__ __host__  inline const Vec3 crossProduct(const Vec3& v1, const Vec3& v2)
{
	return Vec3(
	(v1.y * v2.z) - (v1.z * v2.y),
	(v1.z * v2.x) - (v1.x * v2.z),
	(v1.x * v2.y) - (v1.y * v2.x)
	);	//NOTE: check me

}

	//v1 and v2 unnormalized
 __device__ __host__ inline float angleBetween(Vec3& v1, Vec3& v2)
{
	const float lf = v1.length() * v2.length();

	if(!lf)
		return 1.57079632679489661923f;

	const float dp = dotProduct(v1, v2);

	return acos( dp / lf);
}

 __device__ __host__  inline float angleBetweenNormalized(const Vec3& v1, const Vec3& v2)
{
	const float dp = dotProduct(v1, v2);

	return acos(dp);
}

__device__	__host__ inline float length(const Vec3& v)
{
	return (sqrt(v.x*v.x+v.y*v.y+v.z*v.z));
}
inline bool epsEqual(const Vec3& v1, const Vec3& v2)
{
	const float dp = dotProduct(v1, v2);

	return dp >= 0.99999f;
}
__device__	__host__ inline Vec3 directProduct(const Vec3& v1,const Vec3& v2)
{
	return Vec3(v1.x * v2.x , v1.y * v2.y , v1.z * v2.z);
}
inline std::ostream& operator << (std::ostream& stream, const Vec3& point)
{
	stream << point.x << " ";
	stream << point.y << " ";	
	stream << point.z << " ";

	return stream;
}

inline std::istream& operator >> (std::istream& stream, Vec3& point)
{
	stream >> point.x;
	stream >> point.y;	
	stream >> point.z;

	return stream;
}
