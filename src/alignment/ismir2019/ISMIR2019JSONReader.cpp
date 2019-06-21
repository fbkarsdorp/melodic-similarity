/*
Author: Peter van Kranenburg (peter.van.kranenburg@meertens.knaw.nl)
Copyright 2011 Meertens Institute (KNAW)

This file is part of libmusical.

libmusical is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

libmusical is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with libmusical.  If not, see <http://www.gnu.org/licenses/>.
*/


#include <string>
#include <iostream>
#include <iomanip>
#include <stdexcept>
using namespace std;

#include "ISMIR2019JSONReader.h"
#include "ISMIR2019Symbol.h"
#include "ISMIR2019Sequence.h"
#include "libjson.h"


namespace musical {

ISMIR2019JSONReader::ISMIR2019JSONReader() {
	// TODO Auto-generated constructor stub

}

ISMIR2019JSONReader::~ISMIR2019JSONReader() {
	// TODO Auto-generated destructor stub
}

Sequence* ISMIR2019JSONReader::generateSequence() const {
	//create a new sequence
	ISMIR2019Sequence * nwseq = new ISMIR2019Sequence; //NB ISMIR2019Sequence contains a pitch histogram
	string json_string = source->getJSONString();
	//nwseq->json_string = json_string;
	//cout << json_string << endl;
	JSONNode seq;
	try {
		seq = libjson::parse(json_string);
	} catch (invalid_argument&) {
		std::string errormessage = "Error: Not proper json";
		clog << errormessage << endl;
		throw std::runtime_error(errormessage);
	}
	if ( seq.empty() ) {
		std::string errormessage = "Error: Top node has no children";
		clog << errormessage << endl;
		throw std::runtime_error(errormessage);
	}
	//the name should be at 'top-level'
	JSONNode::const_iterator i1 = seq.begin();
	//cout << i1->size() << endl;
	nwseq->setName(i1->name());
	//cout << nwseq->name << endl;
	i1 = seq.begin()->find("symbols");
	if ( i1 == (seq.begin())->end() ) {
		std::string errormessage = "Error: No symbols in json " + nwseq->getName();
		clog << errormessage << endl;
		throw std::runtime_error(errormessage);
	}
	int size = i1->size();
	//cout << "Size: " << size << endl;
	for( int ix=0; ix<size; ix++) {
		ISMIR2019Symbol* s = new ISMIR2019Symbol();
		try
		{
			s->contour3 = i1->at(ix).at("contour3").as_string();
			s->tonic = i1->at(ix).at("tonic").as_string();
			s->pitch = i1->at(ix).at("pitch").as_string();
			s->imacontour = i1->at(ix).at("imacontour").as_string();
			s->beat_fraction_str = i1->at(ix).at("beat_fraction_str").as_string();
			s->mode = i1->at(ix).at("mode").as_string();
			s->metriccontour = i1->at(ix).at("metriccontour").as_string();
			s->timesignature = i1->at(ix).at("timesignature").as_string();
			s->contour5 = i1->at(ix).at("contour5").as_string();
			s->beat_str = i1->at(ix).at("beat_str").as_string();

			s->beatstrength = i1->at(ix).at("beatstrength").as_float();
			s->phrasepos = i1->at(ix).at("phrasepos").as_float();
			s->beat = i1->at(ix).at("beat").as_float();
			s->imaweight = i1->at(ix).at("imaweight").as_float();
			s->duration = i1->at(ix).at("duration").as_float();
			s->IOR = i1->at(ix).at("IOR").as_float();

			s->diatonicinterval = i1->at(ix).at("diatonicinterval").as_int();
			s->scaledegree = i1->at(ix).at("scaledegree").as_int();
			s->chromaticinterval = i1->at(ix).at("chromaticinterval").as_int();
			s->pitch40 = i1->at(ix).at("pitch40").as_int();
			s->midipitch = i1->at(ix).at("midipitch").as_int();
			s->pitch12 = i1->at(ix).at("midipitch").as_int();

		}
		catch (out_of_range&)
		{
			std::string errormessage = "Error: symbols in " + nwseq->getName() ;
			clog << errormessage << endl;
			throw std::runtime_error(errormessage);
		}
		//check whether id is present. If not take index as id
		//string id = i1->at(ix).at("id").as_string();
		//s->strings["id"] = id;
		nwseq->addSymbol(s);
		//cout << "symbol: " << s->pitch40 << " - " << s->phrasepos << " - " << s->IMA << endl;
	}

	//cout << "size: " << size << endl;

	if ( size > 1) {
		//set next and previous
		nwseq->getSymbolAt(0)->setNext(nwseq->getSymbolAt(1));
		nwseq->getSymbolAt(0)->setPrevious(NULL);
		for( unsigned int i = 1; i<nwseq->size()-1; i++) {
			nwseq->getSymbolAt(i)->setPrevious(nwseq->getSymbolAt(i-1));
			nwseq->getSymbolAt(i)->setNext(nwseq->getSymbolAt(i+1));
		}
		nwseq->getSymbolAt(nwseq->size()-1)->setPrevious( nwseq->getSymbolAt(nwseq->size()-2) );
		nwseq->getSymbolAt(nwseq->size()-1)->setNext(NULL);
	} else if (size == 1) {
		nwseq->getSymbolAt(0)->setNext(NULL);
		nwseq->getSymbolAt(0)->setPrevious(NULL);
	}

	/*
	if ( size > 0 ) {
		//set songposition as fraction of onset of last note
		int minonset = static_cast<ISMIR2019Symbol*>(nwseq->getSymbolAt(0))->onset;
		int maxonset = static_cast<ISMIR2019Symbol*>(nwseq->getSymbolAt(nwseq->size()-1))->onset;
		int length = maxonset - minonset;
		for (unsigned int ix=0; ix<nwseq->size(); ix++) {
			ISMIR2019Symbol * sym = static_cast<ISMIR2019Symbol*>(nwseq->getSymbolAt(ix));
			sym->songpos = float((sym->onset)-minonset) / float(length);
		}
	}
	*/

	if ( size > 0) {

		//read the normalized pitch40 histogram
		//or create it if it is not present in JSON

		for(int i=0; i<200; i++) {
			nwseq->pitchHistogram[i] = 0.0;
		}
		i1 = seq.begin()->find("pitch40histogram");
		if ( i1 == (seq.begin())->end() ) { //not present
			//cout << "Creating pitch histogram for " << nwseq->getName() << endl;
			// count pitches
			int count = 0;
			for (unsigned int ix=0; ix<nwseq->size(); ix++) {
				int indx = static_cast<ISMIR2019Symbol*>(nwseq->getSymbolAt(ix))->pitch40;
				//cout << "index: " << indx << endl;
				if ( indx >= 40 && indx < 240) { nwseq->pitchHistogram[indx - 40] += 1.0; count++; }
			}
			// normalize
			for(int i=0; i<200; i++) {
				nwseq->pitchHistogram[i] = nwseq->pitchHistogram[i] / (double)count;
			}
		} else { //present
			//cout << "Reading pitch histogram for " << nwseq->getName() << endl;
			size = i1->size();
			int pitch = 0;
			for ( int ix=0; ix<size; ix++) {
				pitch = i1->at(ix).at("pitch40").as_int() - 40;
				if (pitch >=0 && pitch < 200 )
					nwseq->pitchHistogram[pitch] = i1->at(ix).at("value").as_float();
			}
		}

		/*
		for(int i=0; i<200; i++) {
			cout << setw(4) << i+40 << " : " << nwseq->pitchHistogram[i] << endl;
		}
		*/

	} // if size > 0

	return nwseq;
}

}
