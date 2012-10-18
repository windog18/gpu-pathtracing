////////////////////////////////////////////////////////////////////////////////////////////////////
// OBJCORE: A Simple Obj Library
// by Yining Karl Li
// Modified by Yulong Shi
// objloader.cpp
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "objloader.h"
#include <iomanip>
#include <sstream>
#include <iostream>
#include <fstream>
#include <string.h>
#include <sstream>
#include "../glm/glm.hpp" 

using namespace std;

objLoader::objLoader(string filename, obj* newMesh){

	geomesh = newMesh;
	cout << "Loading OBJ File: " << filename << endl;
	ifstream fp_in;
	char * fname = (char*)filename.c_str();
	fp_in.open(fname);
	int materialID = 0;
	if(fp_in.is_open()){
        while (fp_in.good() ){
			string line;
            getline(fp_in,line);
            if(line.empty()){
                line="42";
            }
			istringstream liness(line);
			if(line.substr(0,6)=="mtllib"){
				stringstream strStream;
				strStream<<line;
				string temp,materialName;
				strStream>>temp>>materialName;
				int pos = filename.find_last_of('\\');
				if(pos == -1)
					pos = filename.find_last_of('/');
				if(pos == -1)
					pos = 0;
				materialName = filename.substr(0,pos+1)+materialName;
				geomesh->loadMaterial(materialName);
			}else if(line.substr(0,6)=="usemtl"){
				stringstream strStream;
				strStream<<line;
				string temp,fileName;
				strStream>>temp>>fileName;
				if(geomesh->materialNameToID[fileName]!=0){
				  materialID = geomesh->materialNameToID[fileName];
				}
			}else if(line[0]=='v' && line[1]=='t'){
				string v;
				string x;
				string y;
				string z;
				/*getline(liness, v, ' ');
				getline(liness, x, ' ');
				getline(liness, y, ' ');
				getline(liness, z, ' ');*/
				stringstream strStream;
				strStream<<line;
				strStream>>v>>x>>y>>z;
				geomesh->addTextureCoord(glm::vec3(::atof(x.c_str()), ::atof(y.c_str()), ::atof(z.c_str())));
			}else if(line[0]=='v' && line[1]=='n'){
				string v;
				string x;
				string y;
				string z;
				/*getline(liness, v, ' ');
				getline(liness, x, ' ');
				getline(liness, y, ' ');
				getline(liness, z, ' ');*/
				stringstream strStream;
				strStream<<line;
				strStream>>v>>x>>y>>z;
				geomesh->addNormal(glm::vec3(::atof(x.c_str()), ::atof(y.c_str()), ::atof(z.c_str())));
			}else if(line[0]=='v'){
				string v;
				string x;
				string y;
				string z;
				stringstream strStream;
				strStream<<line;
				strStream>>v>>x>>y>>z;
				/*getline(liness, v, ' ');
				getline(liness, x, ' ');
				getline(liness, y, ' ');
				getline(liness, z, ' ');*/
				geomesh->addPoint(glm::vec3(::atof(x.c_str()), ::atof(y.c_str()), ::atof(z.c_str())));
			}else if(line[0]=='f'){
				string v;
				getline(liness, v, ' ');
				string delim1 = "//";
				string delim2 = "/";
				if(std::string::npos != line.find("//")){
					//std::cout << "Vertex-Normal Format" << std::endl;
					vector<int> pointList;
					vector<int> normalList;
					while(getline(liness, v, ' ')){
						istringstream facestring(v);
						string f;
						getline(facestring, f, '/');
						pointList.push_back(::atof(f.c_str())-1);

						getline(facestring, f, '/');
						getline(facestring, f, ' ');
						normalList.push_back(::atof(f.c_str())-1);

					}
					geomesh->addFace(pointList,materialID);
					geomesh->addFaceNormal(normalList);
				}else if(std::string::npos != line.find("/")){
					vector<int> pointList;
					vector<int> normalList;
					vector<int> texturecoordList;
					while(getline(liness, v, ' ')){
						istringstream facestring(v);
						string f;
						int i=0;
						while(getline(facestring, f, '/')){
							if(i==0){
								pointList.push_back(::atof(f.c_str())-1);
							}else if(i==1){
								texturecoordList.push_back(::atof(f.c_str())-1);
							}else if(i==2){
								normalList.push_back(::atof(f.c_str())-1);
							}
							i++;
						}
					}
					geomesh->addFace(pointList,materialID);
					geomesh->addFaceNormal(normalList);
					geomesh->addFaceTexture(texturecoordList);
				}else{
					string v;
					vector<int> pointList;
					while(getline(liness, v, ' ')){
						pointList.push_back(::atof(v.c_str())-1);
					}
					geomesh->addFace(pointList,materialID);
					//std::cout << "Vertex Format" << std::endl;
				}
			}
		}
		if(geomesh->getFaceNormals()->size()==0){
			cout << "normal not find,will count it automatically"<<endl;
			vector<vector<int>> *faces = geomesh->getFaces();
			vector<glm::vec4> *points = geomesh->getPoints();
			vector<int> normals;
			for(int i = 0;i<faces->size();i++){
				const vector<int> *face = &(*faces)[i];
				for(int j = 0;j<face->size();j++){
					glm::vec4 v1 = (*points)[(*face)[j]] - (*points)[(*face)[(j - 1 + face->size())%face->size()]];
					glm::vec4 v2 = (*points)[(*face)[j]] - (*points)[(*face)[(j + 1)%face->size()]];
					v1 = glm::normalize(v1);
					v2 = glm::normalize(v2);
					glm::vec3 n = glm::cross(glm::vec3(v2),glm::vec3(v1));
					geomesh->addNormal(n);
					normals.push_back(geomesh->getNormals()->size()-1);
				}
				geomesh->addFaceNormal(normals);
			}
			cout << "count normal finished..."<<endl;
		}
		cout << "Loaded " << geomesh->getFaces()->size() << " faces, " << geomesh->getPoints()->size() << " vertices from " << filename << endl;
	}else{
        cout << "ERROR: " << filename << " could not be found" << endl;
    }
}

objLoader::~objLoader(){
}

obj* objLoader::getMesh(){
	return geomesh;
}
