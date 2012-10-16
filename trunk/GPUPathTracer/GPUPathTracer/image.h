#ifndef IMAGE_H
#define IMAGE_H

#include "color.h"
#include <ctime>

// TODO: Make this a class!!! (It's no longer used in the .cu file!)

class Image {
public:
	Image(int width,int height){
		pixels = new Color[width*height];
		numPixels = width * height;
		passCounter = 0;
	}
	Image(){
		pixels=NULL;
		numPixels=0;
		width = 0;
		height = 0;
		passCounter = 0;
	}
	~Image(){
		delete pixels;
	}
	Color* pixels;
	int width;
	int height;
	int numPixels;
	int passCounter;
	clock_t startClock;

};



Image* newImage(int width, int height);
void deleteImage(Image* image);
int pixelIndexRowColumn(Image* image, int i, int j);
Color& getPixelRowColumn(Image* image, int i, int j);
void setPixelRowColumn(Image* image, int i, int j, Color c);
float getSecondsElapsed(Image* image);
float getFramesPerSecond(Image* image);

#endif // IMAGE_H