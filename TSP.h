#ifndef TSP_H
#define TSP_H

class TSP{
	public:
		float** list;
		TSP(char *);
		float** GetNodeListFromTSPFile(char*);
};

#endif
