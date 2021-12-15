/*
   This software is based on the code by Xiaohu Lu
   https://arxiv.org/abs/1901.02532
   https://github.com/xiaohulugo/3DLineDetection
*/

#include <stdio.h>
#include <fstream>
#include <liblas\liblas.hpp>

#include "LineDetection3D.h"
#include "nanoflann.hpp"
#include "utils.h"
#include "Timer.h"

using namespace cv;
using namespace std;
using namespace nanoflann;

const int MAXNUMPTS = 100000000;

void readDataFromLASFile(std::string filepath, PointCloud<double> &cloud)
{
	cloud.pts.reserve(MAXNUMPTS);

	cout << "READING DATA (LAS/LAZ FORMAT)--------------------------------------" << endl;

	std::ifstream lasReader( filepath, std::ios::in | std::ios::binary );
	std::vector<cv::Point3d> lidarPoints;
	double x = 0, y = 0, z = 0;
	liblas::ReaderFactory f;
	liblas::Reader reader = f.CreateWithStream( lasReader );

	liblas::Header const& header = reader.GetHeader();
	std::cout << "Compressed   : " << ((header.Compressed() == true) ? "true" : "false") << endl;
	std::cout << "Signature    : " << header.GetFileSignature() << endl;
	std::cout << "Points count : " << header.GetPointRecordsCount() << endl;

	while (reader.ReadNextPoint())
	{
		liblas::Point const& p = reader.GetPoint();
		cloud.pts.push_back(PointCloud<double>::PtData(p.GetX(), p.GetY(), p.GetZ()));
	}
	lasReader.close();

	std::cout << "Total num of points: " << cloud.pts.size() << endl << endl;

}

void readDataFromFile( std::string filepath, PointCloud<double> &cloud )
{
	cloud.pts.reserve(MAXNUMPTS);

	// read in point data
	cout << "READING DATA (TXT FORMAT)------------------------------------------" << endl;

	std::ifstream ptReader( filepath );
	std::vector<cv::Point3d> lidarPoints;
	double x = 0, y = 0, z = 0;
	string line;
	if( ptReader.is_open() )
	{
		while (getline(ptReader, line))
		{
			stringstream(line) >> x >> y >> z;
			cloud.pts.push_back(PointCloud<double>::PtData(x, y, z));
		}
		ptReader.close();
	}

	std::cout << "Total num of points: " << cloud.pts.size() << endl << endl;
}

void writeOutPlanes( string filePath, std::vector<PLANE> &planes, double scale )
{
	// write out bounding polygon result
	string fileEdgePoints = filePath + "planes.txt";
	cout << "Writing: " << fileEdgePoints << endl;
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
	cout << "Writing: " << fileEdgePoints << endl;
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
	string filePoints = filePath + "classified.txt";
	cout << "Writing: " << filePoints << endl;
	FILE *fp2 = fopen(filePoints.c_str(), "w");

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
	cout << "+-----------------------------------------------------------------+" << endl;
	cout << "|             3D SEGMENTS, LINES AND PLANES DETECTION             |" << endl;
	cout << "+-----------------------------------------------------------------+" << endl;
}

void Usage()
{
	cerr << "USE: 3DSegmentation <Filename.txt> [threshold-angle] [threshold-dZ] [supporting-points]" << endl << endl;
	cerr << "+----------------------------------------------------------------------+" << endl;
	cerr << "+--  FileName.txt must contain X Y Z (in that order) in the first      |" << endl;
	cerr << "|    three columes. Any additional columns will be skipped             |" << endl;
	cerr << "|                                                                      |" << endl;
	cerr << "+--  [threshold-angle] is the threshold angle (in degrees!) for the    |" << endl;
	cerr << "|    normals in the segmentation. If not set this option defaults to   |" << endl;
	cerr << "|    15 degrees                                                        |" << endl;
	cerr << "|                                                                      |" << endl;
	cerr << "+--  [threshold-dZ] is threshold for the deviation from the estimated  |" << endl;
	cerr << "|    plane during segmentation. If not set this option defaults to 2.5 |" << endl;
	cerr << "|                                                                      |" << endl;
	cerr << "+--  [supporting-points] is the number of supporting points per plane  |" << endl;
	cerr << "|    for the segementation. If not set this option defaults to 20      |" << endl;
	cerr << "|                                                                      |" << endl;
	cerr << "|    NOTE: EITHER ALL OR NONE OF THE OPTIONS NEED TO BE SET!          |" << endl;
	cerr << "+---------------------------------------------------------------------GV" << endl;
}

bool lasOrNot(string filename)
{
	string UP = filename;
	for_each(UP.begin(), UP.end(), [](char& c) {
		c = ::toupper(c);
	});
	string EXT = UP.substr(UP.find_last_of(".") + 1, 3);
	if (EXT == "LAS" || EXT == "LAZ") return true;

	return false;
}

bool does_file_exist(string fileName)
{
	std::ifstream infile(fileName);
	return infile.good();
}

void main(int argc, char** argv) 
{
	int k = 20;
	double thAngle = 15.0 / 180.0*CV_PI;
	double thRegionSize = 600000;
	double a = 1.4826;
	double thRz = 2.5;

	//cmdl handling
	if (argc != 2 && argc != 5) {
		Usage();
		exit(99);
	}
	if (argc == 5) {
		thAngle = stod(argv[2]) / 180.0*CV_PI;
		thRz = stod(argv[3]);
		k = stoi(argv[4]);
	}

	//set filenames
	//DEB: fileData = "C:\\Users\\VestjensGerbrand\\OneDrive - Kadaster\\Kwaliteit_3D_Modellen\\TestPlaneSegmentation\\C_05HN2_Selection.txt";  //"D://Facade//data.txt";
	string fileData = argv[1];
	string fileOut = fileData.substr(0, fileData.find_last_of(".")) + "_";

	//check if inputfile exists...
	if (!does_file_exist(fileData)) {
		cerr << "** File does not exist: " << fileData << endl;
		exit(99);
	}

	bool isLas = lasOrNot(fileData);

	// header
	writeHeader();
	cout << "OPTIONS------------------------------------------------------------" << endl;
	cout << "Inputfile         : " << fileData << endl;
	cout << "Output prefix     : " << fileOut << endl;
	cout << "Normals threshold : " << thAngle * 180 / CV_PI << " degrees." << endl;
	cout << "Delta-Z threshold : " << thRz << endl;
	cout << "Supporting points : " << k << " points." << endl;
	cout << "Amplification fct : " << a << endl;
	cout << "Regionsize thresh : " << thRegionSize << endl;
	cout << endl;

	// read in data
	PointCloud<double> pointData; 
	if (isLas) {
		readDataFromLASFile(fileData, pointData);
	}
	else {
		readDataFromFile(fileData, pointData);
	}

	LineDetection3D detector;
	std::vector<PLANE> planes;
	std::vector<std::vector<cv::Point3d> > lines;
	std::vector<double> ts;
	std::vector<std::vector<int>> regions;

	//run detector
	detector.run( pointData, k, planes, lines, ts, regions, thAngle, a, thRz, thRegionSize);
	cout << "RESULTS------------------------------------------------------------" << endl;
	cout << "Regions number : " << regions.size() << endl;
	cout << "Lines number   : " << lines.size() << endl;
	cout << "Planes number  : " << planes.size() << endl << endl;
	
	//create output
	cout << "WRITING OUTPUT-----------------------------------------------------" << endl;
	writeOutPlanes( fileOut, planes, detector.scale );
	writeOutLines( fileOut, lines, detector.scale );
	writeOutClassifiedData( fileOut, pointData, regions );

	//finish
	cout << endl << "Done!" << endl;

}