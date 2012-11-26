#ifndef TSP_H
#define TSP_H


class TSP{
private:
	TSPList a;
public:
	TSP(char* filename);
	TSPList GetNodeListFromTSPFile(char* filename);
	TSPList list();
	void list(TSPList list);
};

#endif

