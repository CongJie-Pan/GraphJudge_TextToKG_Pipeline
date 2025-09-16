const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const multer = require('multer'); // For file upload handling

// Create Express application
const app = express();
const PORT = process.env.PORT || 3000;

// Directory to store uploaded graph files
const GRAPHS_DIR = path.join(__dirname, 'graphs');

// Ensure graphs directory exists
if (!fs.existsSync(GRAPHS_DIR)) {
    fs.mkdirSync(GRAPHS_DIR, { recursive: true });
    console.log('📁 Created graphs directory');
}

// Configure multer for file uploads
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, GRAPHS_DIR);
    },
    filename: (req, file, cb) => {
        // Generate unique filename with timestamp
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const ext = path.extname(file.originalname);
        const name = path.basename(file.originalname, ext);
        cb(null, `${name}_${timestamp}${ext}`);
    }
});

const upload = multer({
    storage: storage,
    fileFilter: (req, file, cb) => {
        // Only accept JSON files
        if (file.mimetype === 'application/json' || path.extname(file.originalname) === '.json') {
            cb(null, true);
        } else {
            cb(new Error('Only JSON files are allowed'), false);
        }
    },
    limits: {
        fileSize: 50 * 1024 * 1024 // 50MB limit
    }
});

// Enable CORS for all origins
app.use(cors());

// Parse JSON bodies
app.use(express.json());

// Serve static files from current directory
app.use(express.static(__dirname));

// Helper function to validate knowledge graph JSON structure
function validateGraphData(data) {
    if (!data || typeof data !== 'object' || Array.isArray(data)) {
        return { valid: false, error: "The root of the JSON file must be an object (e.g., {...})." };
    }

    // Validate 'entities'
    if (!('entities' in data)) {
        return { valid: false, error: "The JSON object is missing the required 'entities' key." };
    }
    if (!Array.isArray(data.entities)) {
        return { valid: false, error: "The 'entities' key must correspond to an array (e.g., \"entities\": [...])." };
    }
    const firstNonStringEntity = data.entities.find(e => typeof e !== 'string');
    if (firstNonStringEntity !== undefined) {
        return { valid: false, error: `All items in the 'entities' array must be strings. Found an item of type ${typeof firstNonStringEntity}: ${JSON.stringify(firstNonStringEntity)}` };
    }

    // Validate 'relationships'
    if (!('relationships' in data)) {
        return { valid: false, error: "The JSON object is missing the required 'relationships' key." };
    }
    if (!Array.isArray(data.relationships)) {
        return { valid: false, error: "The 'relationships' key must correspond to an array (e.g., \"relationships\": [...])." };
    }
    const firstNonStringRelationship = data.relationships.find(r => typeof r !== 'string');
    if (firstNonStringRelationship !== undefined) {
         return { valid: false, error: `All items in the 'relationships' array must be strings. Found an item of type ${typeof firstNonStringRelationship}: ${JSON.stringify(firstNonStringRelationship)}` };
    }
    const improperlyFormattedRelationship = data.relationships.find(r => (r.split(' - ').length < 2 && r.split('-').length < 2) ); // check both - and ' - '
    if (improperlyFormattedRelationship !== undefined) {
        return { valid: false, error: `Relationships must follow the 'Source - Relation - Target' format. The relationship "${improperlyFormattedRelationship}" is not formatted correctly.` };
    }

    return { valid: true, error: null };
}

// Helper function to get file metadata
function getFileMetadata(filePath) {
    const stats = fs.statSync(filePath);
    const filename = path.basename(filePath);
    
    try {
        const rawData = fs.readFileSync(filePath, 'utf8');
        const jsonData = JSON.parse(rawData);
        const validation = validateGraphData(jsonData);
        
        return {
            filename,
            size: stats.size,
            created: stats.birthtime,
            modified: stats.mtime,
            entities: jsonData.entities ? jsonData.entities.length : 0,
            relationships: jsonData.relationships ? jsonData.relationships.length : 0,
            valid: validation.valid,
            error: validation.error
        };
    } catch (error) {
        return {
            filename,
            size: stats.size,
            created: stats.birthtime,
            modified: stats.mtime,
            entities: 0,
            relationships: 0,
            valid: false,
            error: error.message
        };
    }
}

// API endpoint to list available graph files
app.get('/api/graph-files', (req, res) => {
    try {
        const files = fs.readdirSync(GRAPHS_DIR)
            .filter(file => file.endsWith('.json'))
            .map(file => {
                const filePath = path.join(GRAPHS_DIR, file);
                return getFileMetadata(filePath);
            })
            .sort((a, b) => new Date(b.modified) - new Date(a.modified)); // Sort by newest first
        
        // Also check for the original file in parent directory
        const originalPath = path.join(__dirname, '..', 'final_results_streamlined_20250619_133346.json');
        if (fs.existsSync(originalPath)) {
            files.unshift({
                ...getFileMetadata(originalPath),
                filename: 'final_results_streamlined_20250619_133346.json (original file)',
                isOriginal: true
            });
        }
        
        res.json({ files });
    } catch (error) {
        console.error('Error listing graph files:', error);
        res.status(500).json({ error: 'Failed to list graph files', message: error.message });
    }
});

// API endpoint to upload a new graph file
app.post('/api/upload-graph', upload.single('graph'), (req, res) => {
    // Wrapped in outer try-catch for general errors
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No file uploaded or file is not JSON.' });
        }
        
        const filePath = req.file.path;
        let jsonData;

        // Step 1: Read and parse the file
        try {
            const rawData = fs.readFileSync(filePath, 'utf8');
            jsonData = JSON.parse(rawData);
        } catch (parseError) {
            fs.unlinkSync(filePath); // Delete invalid file
            console.error(`❌ Invalid JSON in uploaded file: ${req.file.filename}`, parseError.message);
            return res.status(400).json({ 
                error: 'Invalid JSON Syntax', 
                details: `The uploaded file contains a syntax error and could not be parsed. Please validate the JSON structure. Error: ${parseError.message}`
            });
        }
        
        // Step 2: Validate the graph structure
        const validation = validateGraphData(jsonData);
        
        if (!validation.valid) {
            fs.unlinkSync(filePath); // Delete invalid file
            console.error(`❌ Invalid graph structure in ${req.file.filename}: ${validation.error}`);
            return res.status(400).json({ 
                error: 'Invalid Graph File Format', 
                details: validation.error // The detailed message from validateGraphData
            });
        }
        
        // Step 3: If valid, process and return metadata
        const metadata = getFileMetadata(filePath);
        
        res.json({
            message: 'File uploaded successfully',
            file: metadata
        });
        
        console.log(`📤 New graph file uploaded: ${req.file.filename}`);
        
    } catch (error) {
        // General catch for other errors (e.g., file system issues)
        console.error('An unexpected error occurred during file upload:', error);
        
        // Clean up file if it exists
        if (req.file && fs.existsSync(req.file.path)) {
            fs.unlinkSync(req.file.path);
        }
        
        res.status(500).json({ 
            error: 'Failed to process uploaded file', 
            message: error.message 
        });
    }
});

// API endpoint to delete a graph file
app.delete('/api/graph-files/:filename', (req, res) => {
    try {
        const filename = req.params.filename;
        const filePath = path.join(GRAPHS_DIR, filename);
        
        if (!fs.existsSync(filePath)) {
            return res.status(404).json({ error: 'File not found' });
        }
        
        fs.unlinkSync(filePath);
        res.json({ message: 'File deleted successfully' });
        
        console.log(`🗑️ Graph file deleted: ${filename}`);
        
    } catch (error) {
        console.error('Error deleting file:', error);
        res.status(500).json({ error: 'Failed to delete file', message: error.message });
    }
});

// Modified API endpoint to serve the knowledge graph data (now supports file selection)
app.get('/api/graph-data', (req, res) => {
    try {
        const selectedFile = req.query.file; // Allow file selection via query parameter
        let dataPath;
        
        console.log('='.repeat(60));
        console.log(`📡 API Request - Get graph data`);
        console.log(`🕐 Request time: ${new Date().toLocaleString('zh-TW')}`);
        console.log(`📁 Requested file: ${selectedFile || 'default file'}`);
        
        if (selectedFile) {
            // If specific file requested, look for it in graphs directory
            if (selectedFile.includes('original') || selectedFile.includes('final_results_streamlined')) {
                // Handle original file
                dataPath = path.join(__dirname, '..', 'final_results_streamlined_20250619_133346.json');
                console.log(`📂 Loading original file: ${dataPath}`);
            } else {
                dataPath = path.join(GRAPHS_DIR, selectedFile);
                console.log(`📂 Loading uploaded file: ${dataPath}`);
            }
        } else {
            // Default to original file if no specific file requested
            dataPath = path.join(__dirname, '..', 'final_results_streamlined_20250619_133346.json');
            console.log(`📂 Loading default file: ${dataPath}`);
        }
        
        // Check if file exists
        if (!fs.existsSync(dataPath)) {
            const errorMessage = `❌ File does not exist: ${dataPath}`;
            console.error(errorMessage);
            
            // If the requested file doesn't exist (default or specific),
            // always return an empty graph with an informative message instead of a 404 error.
            // This prevents the frontend from crashing and informs the user.
            const emptyGraphData = {
                entities: [],
                relationships: [],
                source_file: 'No file loaded',
                timestamp: new Date().toISOString(),
                processing_time: Date.now(),
                message: `Graph file not found: "${path.basename(dataPath)}"`,
                error: errorMessage,
                empty_graph: true
            };
            
            console.log(`🔄 Requested file not found, returning empty graph structure`);
            console.log(`✅ Returning empty graph structure`);
            console.log('='.repeat(60));
            
            // Always return 200 OK with the empty graph payload
            return res.status(200).json(emptyGraphData);
        }
        
        console.log(`✅ File exists, starting to read...`);

        // Read and parse the JSON file
        const rawData = fs.readFileSync(dataPath, 'utf8');
        console.log(`📄 File size: ${(rawData.length / 1024).toFixed(2)} KB`);
        
        const jsonData = JSON.parse(rawData);
        console.log(`🔍 JSON parsing completed`);
        
        // Validate file structure
        const validation = validateGraphData(jsonData);
        if (!validation.valid) {
            console.error(`❌ File format validation failed: ${validation.error}`);
            return res.status(400).json({
                error: 'Invalid graph file structure',
                details: validation.error
            });
        }
        
        console.log(`✅ File format validation passed`);
        
        // Fix data inconsistencies: ensure all entities mentioned in relationships exist in entities list
        const entitySet = new Set(jsonData.entities);
        const missingEntities = new Set();
        
        console.log(`🔧 Checking data consistency...`);
        console.log(`📊 Original entity count: ${jsonData.entities.length}`);
        console.log(`📊 Relationship count: ${jsonData.relationships.length}`);
        
        jsonData.relationships.forEach(rel => {
            const parts = rel.split(' - ');
            if (parts.length >= 3) {
                const source = parts[0];
                const target = parts[parts.length - 1];
                
                if (!entitySet.has(source)) {
                    missingEntities.add(source);
                }
                if (!entitySet.has(target)) {
                    missingEntities.add(target);
                }
            }
        });
        
        // Add missing entities to the entities list
        if (missingEntities.size > 0) {
            console.log(`🔧 Found ${missingEntities.size} missing entities, auto-fixing:`);
            missingEntities.forEach(entity => {
                console.log(`   + Adding entity: ${entity}`);
                jsonData.entities.push(entity);
            });
        } else {
            console.log(`✅ No fixes needed, data consistency is good`);
        }
        
        const finalStats = {
            entities: jsonData.entities.length,
            relationships: jsonData.relationships.length,
            fixes: missingEntities.size,
            file: path.basename(dataPath)
        };
        
        console.log(`📊 Final statistics:`);
        console.log(`   📍 Entity count: ${finalStats.entities}`);
        console.log(`   🔗 Relationship count: ${finalStats.relationships}`);
        console.log(`   🔧 Fixed items: ${finalStats.fixes}`);
        console.log(`   📁 Source file: ${finalStats.file}`);
        
        // Prepare response data
        const responseData = {
            ...jsonData,
            fixes_applied: missingEntities.size,
            missing_entities_found: Array.from(missingEntities),
            source_file: path.basename(dataPath),
            timestamp: new Date().toISOString(),
            processing_time: Date.now()
        };
        
        console.log(`✅ Successfully loaded graph data: ${finalStats.file} - ${finalStats.entities} entities, ${finalStats.relationships} relationships`);
        console.log(`📤 Sending response data...`);
        
        // Send the data with proper headers
        res.setHeader('Content-Type', 'application/json');
        res.setHeader('Cache-Control', 'no-cache');
        res.json(responseData);
        
        console.log(`✅ Response sent successfully`);
        console.log('='.repeat(60));
        
    } catch (error) {
        console.log('='.repeat(60));
        console.error('❌ Error occurred while loading graph data:');
        console.error(`🚨 Error type: ${error.name}`);
        console.error(`📝 Error message: ${error.message}`);
        console.error(`📍 Error stack:`);
        console.error(error.stack);
        console.log('='.repeat(60));
        
        res.status(500).json({ 
            error: 'Failed to load graph data',
            message: error.message,
            details: 'Error occurred while reading knowledge graph data',
            timestamp: new Date().toISOString()
        });
    }
});

// Root route - serve the main visualizer
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

// Route for simple viewer
app.get('/simple', (req, res) => {
    res.sendFile(path.join(__dirname, 'simple_graph_viewer.html'));
});

// Route for full-featured viewer
app.get('/full', (req, res) => {
    res.sendFile(path.join(__dirname, 'knowledge_graph_visualizer.html'));
});

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({ 
        status: 'healthy', 
        timestamp: new Date().toISOString(),
        message: 'Server is running normally'
    });
});

// Error handling middleware
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).json({ 
        error: 'Something went wrong!',
        message: 'Internal server error'
    });
});

// Handle 404 errors
app.use((req, res) => {
    res.status(404).json({ 
        error: 'Route not found',
        message: 'Requested page not found'
    });
});

// Start the server
app.listen(PORT, () => {
    console.log('='.repeat(60));
    console.log('🚀 Knowledge Graph Visualization Server Started');
    console.log('='.repeat(60));
    console.log(`📍 Server address: http://localhost:${PORT}`);
    console.log(`📊 Main page: http://localhost:${PORT}/`);
    console.log(`🔍 Simple version: http://localhost:${PORT}/simple`);
    console.log(`⚡ Full version: http://localhost:${PORT}/full`);
    console.log(`📡 API endpoint: http://localhost:${PORT}/api/graph-data`);
    console.log(`💚 Health check: http://localhost:${PORT}/health`);
    console.log('='.repeat(60));
    console.log('💡 Tip: Press Ctrl+C to stop server');
    console.log('');
});

// Graceful shutdown
process.on('SIGINT', () => {
    console.log('\n👋 Shutting down server...');
    process.exit(0);
});

process.on('SIGTERM', () => {
    console.log('\n👋 Shutting down server...');
    process.exit(0);
}); 