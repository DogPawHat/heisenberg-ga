#ifndef TSP_H
#define TSP_H

template <typename Iterator>
class TSP{
	private:
		double** a;
	public:
		TSP(char *);
		double** GetNodeListFromTSPFile(char*);
		int parseDimentionSection(Iterator&, Iterator);
		double** parseAdjacencyList(Iterator&, Iterator, int);
		double** list(){return this->a;}
		void list(double** list){this->a = list;}
};
#endif
