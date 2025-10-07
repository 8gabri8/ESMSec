#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstring>
#include <expat.h>

struct ClusterData {
    std::string id;
    std::string name;
    std::vector<std::string> human_proteins;
    std::string current_dbref_id;
    bool current_is_uniprotkb = false;
    bool found_human = false;
    
    void reset() {
        id.clear();
        name.clear();
        human_proteins.clear();
        current_dbref_id.clear();
        current_is_uniprotkb = false;
        found_human = false;
    }
};

ClusterData current_cluster;
std::ofstream output_file;
long entry_count = 0;
long human_cluster_count = 0;
std::string current_text;
std::string output_buffer;

// Preallocate buffer to reduce reallocations
constexpr size_t OUTPUT_BUFFER_SIZE = 1024 * 1024 * 10; // 10MB

void flush_output() {
    if (!output_buffer.empty()) {
        output_file.write(output_buffer.c_str(), output_buffer.size());
        output_buffer.clear();
    }
}

void XMLCALL startElement(void *userData, const char *name, const char **atts) {
    // Fast string comparison using first character check
    char first = name[0];
    
    if (first == 'e' && strcmp(name, "entry") == 0) {
        current_cluster.reset();
        for (int i = 0; atts[i]; i += 2) {
            if (atts[i][0] == 'i' && strcmp(atts[i], "id") == 0) {
                current_cluster.id = atts[i + 1];
                break;
            }
        }
    }
    else if (first == 'd' && strcmp(name, "dbReference") == 0) {
        current_cluster.current_dbref_id.clear();
        current_cluster.current_is_uniprotkb = false;
        current_cluster.found_human = false;
        
        const char* type_val = nullptr;
        const char* id_val = nullptr;
        
        for (int i = 0; atts[i]; i += 2) {
            if (atts[i][0] == 't' && strcmp(atts[i], "type") == 0) {
                type_val = atts[i + 1];
            }
            else if (atts[i][0] == 'i' && strcmp(atts[i], "id") == 0) {
                id_val = atts[i + 1];
            }
        }
        
        if (type_val && strcmp(type_val, "UniProtKB ID") == 0) {
            current_cluster.current_is_uniprotkb = true;
            if (id_val) {
                current_cluster.current_dbref_id = id_val;
            }
        }
    }
    else if (first == 'p' && strcmp(name, "property") == 0 && current_cluster.current_is_uniprotkb) {
        const char* type_val = nullptr;
        const char* value_val = nullptr;
        
        for (int i = 0; atts[i]; i += 2) {
            if (atts[i][0] == 't' && strcmp(atts[i], "type") == 0) {
                type_val = atts[i + 1];
            }
            else if (atts[i][0] == 'v' && strcmp(atts[i], "value") == 0) {
                value_val = atts[i + 1];
            }
        }
        
        if (type_val && strcmp(type_val, "source organism") == 0 && 
            value_val && strstr(value_val, "Homo sapiens") != nullptr) {
            current_cluster.found_human = true;
        }
    }
    
    current_text.clear();
}

void XMLCALL endElement(void *userData, const char *name) {
    char first = name[0];
    
    if (first == 'n' && strcmp(name, "name") == 0 && current_cluster.name.empty()) {
        current_cluster.name = current_text;
    }
    else if (first == 'd' && strcmp(name, "dbReference") == 0) {
        if (current_cluster.found_human && !current_cluster.current_dbref_id.empty()) {
            current_cluster.human_proteins.push_back(std::move(current_cluster.current_dbref_id));
            current_cluster.current_dbref_id.clear();
        }
    }
    else if (first == 'e' && strcmp(name, "entry") == 0) {
        entry_count++;
        
        if (!current_cluster.human_proteins.empty() && !current_cluster.name.empty()) {
            human_cluster_count++;
            
            // Build CSV line in buffer
            output_buffer += current_cluster.id;
            output_buffer += ",\"";
            output_buffer += current_cluster.name;
            output_buffer += "\",\"";
            
            for (size_t i = 0; i < current_cluster.human_proteins.size(); i++) {
                if (i > 0) output_buffer += "; ";
                output_buffer += current_cluster.human_proteins[i];
            }
            output_buffer += "\"\n";
            
            // Flush buffer periodically
            if (output_buffer.size() > OUTPUT_BUFFER_SIZE) {
                flush_output();
            }
        }
        
        if (entry_count % 10000 == 0) {
            std::cerr << "Processed " << entry_count << " entries, found " 
                      << human_cluster_count << " human clusters...\r" << std::flush;
        }
    }
}

void XMLCALL characterData(void *userData, const char *s, int len) {
    current_text.append(s, len);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input.xml> <output.csv>\n";
        return 1;
    }
    
    const char* input_file = argv[1];
    const char* output_csv = argv[2];
    
    std::ifstream infile(input_file, std::ios::binary);
    if (!infile) {
        std::cerr << "Cannot open input file: " << input_file << "\n";
        return 1;
    }
    
    // Set larger input buffer for ifstream
    constexpr size_t INPUT_BUFFER_SIZE = 1024 * 1024 * 4; // 4MB
    char* file_buffer = new char[INPUT_BUFFER_SIZE];
    infile.rdbuf()->pubsetbuf(file_buffer, INPUT_BUFFER_SIZE);
    
    output_file.open(output_csv, std::ios::binary);
    if (!output_file) {
        std::cerr << "Cannot open output file: " << output_csv << "\n";
        delete[] file_buffer;
        return 1;
    }
    
    // Reserve space for output buffer
    output_buffer.reserve(OUTPUT_BUFFER_SIZE);
    
    output_file << "Cluster_ID,Cluster_Name,Human_Proteins\n";
    
    XML_Parser parser = XML_ParserCreate(NULL);
    XML_SetElementHandler(parser, startElement, endElement);
    XML_SetCharacterDataHandler(parser, characterData);
    
    std::cerr << "Parsing XML file...\n";
    
    constexpr size_t PARSE_BUFFER_SIZE = 1024 * 1024 * 8; // 8MB buffer
    char* buffer = new char[PARSE_BUFFER_SIZE];
    
    while (infile.read(buffer, PARSE_BUFFER_SIZE) || infile.gcount() > 0) {
        if (XML_Parse(parser, buffer, infile.gcount(), infile.eof()) == XML_STATUS_ERROR) {
            std::cerr << "\nParse error at line " << XML_GetCurrentLineNumber(parser) << ": "
                      << XML_ErrorString(XML_GetErrorCode(parser)) << "\n";
            delete[] buffer;
            delete[] file_buffer;
            return 1;
        }
    }
    
    // Final flush
    flush_output();
    
    XML_ParserFree(parser);
    output_file.close();
    infile.close();
    
    delete[] buffer;
    delete[] file_buffer;
    
    std::cerr << "\n\nTotal entries: " << entry_count << "\n";
    std::cerr << "Human clusters: " << human_cluster_count << "\n";
    std::cerr << "Results saved to " << output_csv << "\n";
    
    return 0;
}

// Compile with: g++ -O3 -march=native -o extract_human uniref_extractor.cpp -lexpat
// Run with: ./extract_human uniref50.xml fast_c++_human_clusters.csv