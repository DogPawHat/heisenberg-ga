#ifndef ADD_H
#define ADD_H

class TSP{
	public:
		float** list;
		TSP(char *);
		float** GetNodeListFromTSPFile(string);
};

#endif