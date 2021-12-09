#include <stdio.h>
#include <fstream>

#include "LineDetection3D.h"
#include "nanoflann.hpp"
#include "utils.h"
#include "Timer.h"

using namespace cv;
using namespace std;
using namespace nanoflann;

const int DOTSPERLINE = 20;
const int MAXNUMPTS = 10000000;

void readDataFromFile( std::string filepath, PointCloud<double> &cloud )
{
	cloud.pts.reserve(MAXNUMPTS);
	cout<<"Reading data ..."<<endl;

	// 1. read in point data
	std::ifstream ptReader( filepath );
	std::vector<cv::Point3d> lidarPoints;
	double x = 0, y = 0, z = 0, t_stamp = 0, color = 0;
	double nx, ny, nz;
	int a = 0, b = 0, c = 0; 
	int labelIdx = 0;
	int count = 0;
	int countTotal = 0;
	if( ptReader.is_open() )
	{
		while ( !ptReader.eof() ) 
		{
			//ptReader >> x >> y >> z;


			//ptReader >> x >> y >> z >> a >> b >> c >> labelIdx;
			//ptReader >> x >> y >> z >> a >> b >> c >> color;
			//ptReader >> x >> y >> z >> color >> a >> b >> c;
			//ptReader >> x >> y >> z >> a >> b >> c ;
			ptReader >> x >> y >> z >>t_stamp;
			//ptReader >> x >> y >> z >> color;
			//ptReader >> x >> y >> z >> nx >> ny >> nz;

			cloud.pts.push_back(PointCloud<double>::PtData(x,y,z));

		}
		ptReader.close();
	}

	std::cout << "Total num of points: " << cloud.pts.size() << "\n";
}

void writeOutPlanes( string filePath, std::vector<PLANE> &planes, double scale )
{
	// write out bounding polygon result
	string fileEdgePoints = filePath + "planes.txt";
	FILE *fp2 = fopen( fileEdgePoints.c_str(), "w");
	for (int p=0; p<planes.size(); ++p)
	{
		int R = rand()%255;
		int G = rand()%255;
		int B = rand()%255;

		for (int i=0; i<planes[p].lines3d.size(); ++i)
		{
			for (int j=0; j<planes[p].lines3d[i].size(); ++j)
			{
				cv::Point3d dev = planes[p].lines3d[i][j][1] - planes[p].lines3d[i][j][0];
				double L = sqrt(dev.x*dev.x + dev.y*dev.y + dev.z*dev.z);
				int k = L/(scale/10);

				double x = planes[p].lines3d[i][j][0].x, y = planes[p].lines3d[i][j][0].y, z = planes[p].lines3d[i][j][0].z;
				double dx = dev.x/k, dy = dev.y/k, dz = dev.z/k;
				for ( int j=0; j<k; ++j)
				{
					x += dx;
					y += dy;
					z += dz;

					fprintf( fp2, "%.6lf   %.6lf   %.6lf    ", x, y, z );
					fprintf( fp2, "%d   %d   %d   %d\n", R, G, B, p );
				}
			}
		}
	}
	fclose( fp2 );
}

void writeOutLines( string filePath, std::vector<std::vector<cv::Point3d> > &lines, double scale )
{
	// write out bounding polygon result
	string fileEdgePoints = filePath + "lines.txt";
	FILE *fp2 = fopen( fileEdgePoints.c_str(), "w");
	for (int p=0; p<lines.size(); ++p)
	{
		int R = rand()%255;
		int G = rand()%255;
		int B = rand()%255;

		cv::Point3d dev = lines[p][1] - lines[p][0];
		double L = sqrt(dev.x*dev.x + dev.y*dev.y + dev.z*dev.z);
		int k = L/(scale/10);

		double x = lines[p][0].x, y = lines[p][0].y, z = lines[p][0].z;
		double dx = dev.x/k, dy = dev.y/k, dz = dev.z/k;
		for ( int j=0; j<k; ++j)
		{
			x += dx;
			y += dy;
			z += dz;

			fprintf( fp2, "%.6lf   %.6lf   %.6lf    ", x, y, z );
			fprintf( fp2, "%d   %d   %d   %d\n", R, G, B, p );
		}
	}
	fclose( fp2 );
}

void writeOutClassifiedData(string filePath, PointCloud<double> &data, std::vector<std::vector<int>> &regions)
{
	string fileEdgePoints = filePath + "classified.txt";
	FILE *fp2 = fopen(fileEdgePoints.c_str(), "w");

	for (int i = 0; i < regions.size(); ++i)
	{
		std::vector<int> cur = regions[i];
		for (int j = 0; j < cur.size(); ++j)
		{	
			int p = cur[j];
			fprintf(fp2, "%.3lf   %.3lf   %.3lf   %d\n", data.pts[p].x, data.pts[p].y, data.pts[p].z, i);
		}
	}
	fflush(fp2);
	fclose(fp2);
}

void writeHeader()
{
	cout << "+-----------------+" << endl;
	cout << "| 3D SEGMENTATION |" << endl;
	cout << "+-----------------+" << endl;
}

void Usage()
{
	cerr << "USE: 3DSegmentation <filename.txt>" << endl << endl;
	cerr << "     FileName.txt must contain X Y Z n, " << endl;
	cerr << "     where n is either the GPS-time or any other numerical placeholder" << endl;
}


void main(int argc, char** argv) 
{
	//cmdl handling
	if (argc != 2) {
		Usage();
		exit(99);
	}

	//set filenames
	//DEB: fileData = "C:\\Users\\VestjensGerbrand\\OneDrive - Kadaster\\Kwaliteit_3D_Modellen\\TestPlaneSegmentation\\C_05HN2_Selection.txt";  //"D://Facade//data.txt";
	string fileData = argv[1];
	string fileOut = fileData.substr(0, fileData.find_last_of("\\") + 1) + "SEG_";


	// header
	writeHeader();

	// read in data
	PointCloud<double> pointData; 
	readDataFromFile( fileData, pointData );

	LineDetection3D detector;
	std::vector<PLANE> planes;
	std::vector<std::vector<cv::Point3d> > lines;
	std::vector<double> ts;
	std::vector<std::vector<int>> regions;

	//run detector
	detector.run( pointData, DOTSPERLINE, planes, lines, ts, regions);
	cout << "Results" << endl;
	cout << "* regions number: " << regions.size() << endl;
	cout << "* lines number: " << lines.size() << endl;
	cout << "* planes number: " << planes.size() << endl;
	
	//create output
	cout << endl << "Writing output..." << endl;
	writeOutPlanes( fileOut, planes, detector.scale );
	writeOutLines( fileOut, lines, detector.scale );
	writeOutClassifiedData(fileOut, pointData, regions);

	//finish
	cout << "Done!" << endl;

}