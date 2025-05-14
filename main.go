package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/gofiber/fiber/v2"
	"github.com/jinzhu/gorm"
	_ "github.com/lib/pq"
	"github.com/pgvector/pgvector-go"
)

type Photo struct {
	ID        uint            `gorm:"primary_key"`
	FilePath  string          `gorm:"not null"`
	Embedding pgvector.Vector `gorm:"type:vector(512)"`
}

type PythonOutput struct {
	Success   bool      `json:"success"`
	Embedding []float32 `json:"embedding"`
	Error     string    `json:"error"`
}

var db *gorm.DB

func main() {

	var err error
	db, err = gorm.Open("postgres", "host=localhost user=admin dbname=face_db password=1234 sslmode=disable")
	if err != nil {
		panic("Failed to connect to database")
	}
	defer db.Close()

	db.AutoMigrate(&Photo{})

	app := fiber.New()

	app.Post("/upload", uploadPhoto)
	app.Post("/search", searchPhoto)

	app.Listen(":8080")
}

func extractJSON(output []byte) ([]byte, error) {

	start := bytes.IndexByte(output, '{')
	if start == -1 {
		return nil, fmt.Errorf("no JSON found in output")
	}

	end := bytes.LastIndexByte(output, '}')
	if end == -1 {
		return nil, fmt.Errorf("no JSON found in output")
	}

	return output[start : end+1], nil
}

func runPythonScript(scriptPath string, imagePath string) (PythonOutput, error) {
	var output PythonOutput

	cmd := exec.Command("python3", scriptPath, imagePath)
	rawOutput, err := cmd.CombinedOutput()
	if err != nil {
		return output, fmt.Errorf("python script execution failed: %v", err)
	}

	jsonData, err := extractJSON(rawOutput)
	if err != nil {
		return output, fmt.Errorf("failed to extract JSON from output: %v\nOutput was: %s", err, string(rawOutput))
	}

	if err := json.Unmarshal(jsonData, &output); err != nil {
		return output, fmt.Errorf("failed to parse JSON output: %v\nJSON was: %s", err, string(jsonData))
	}

	return output, nil
}

func uploadPhoto(c *fiber.Ctx) error {
	file, err := c.FormFile("photo")
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "No file uploaded"})
	}

	filePath := filepath.Join("uploads", file.Filename)
	if err := c.SaveFile(file, filePath); err != nil {
		return c.Status(500).JSON(fiber.Map{"error": "Failed to save file"})
	}

	pythonScript := "/home/mca/python-arcface/scripts/arcface_embedding.py"
	output, err := runPythonScript(pythonScript, filePath)
	if err != nil {
		return c.Status(500).JSON(fiber.Map{
			"error":   "Face detection failed",
			"details": err.Error(),
		})
	}

	if !output.Success {
		return c.Status(400).JSON(fiber.Map{
			"error":   "Face detection failed",
			"details": output.Error,
		})
	}

	if len(output.Embedding) == 0 {
		return c.Status(400).JSON(fiber.Map{"error": "No face detected"})
	}

	photo := Photo{
		FilePath:  filePath,
		Embedding: pgvector.NewVector(output.Embedding),
	}
	db.Create(&photo)

	return c.JSON(fiber.Map{"id": photo.ID})
}

func searchPhoto(c *fiber.Ctx) error {
	file, err := c.FormFile("photo")
	if err != nil {
		return c.Status(400).JSON(fiber.Map{
			"status": 0,
			"error":  "No file uploaded",
		})
	}

	tempPath := filepath.Join("uploads", "temp_"+file.Filename)
	if err := c.SaveFile(file, tempPath); err != nil {
		return c.Status(500).JSON(fiber.Map{"error": "Failed to save file"})
	}
	defer os.Remove(tempPath)

	pythonScript := "/home/mca/python-arcface/scripts/arcface_embedding.py"
	output, err := runPythonScript(pythonScript, tempPath)
	if err != nil {
		return c.Status(500).JSON(fiber.Map{
			"status":  0,
			"error":   "Face detection failed",
			"details": err.Error(),
		})
	}

	if !output.Success {
		return c.Status(400).JSON(fiber.Map{
			"status":  0,
			"error":   "Face detection failed",
			"details": output.Error,
		})
	}

	if len(output.Embedding) == 0 {
		return c.Status(400).JSON(fiber.Map{
			"status": 0,
			"error":  "No face detected",
		})
	}

	var results []Photo
	maxDistance := 0.5
	db.Raw(`
		SELECT id, file_path
		FROM photos
		WHERE embedding <=> ? < ?
		ORDER BY embedding <=> ?
		LIMIT 10
	`, pgvector.NewVector(output.Embedding), maxDistance, pgvector.NewVector(output.Embedding)).Scan(&results)

	if len(results) == 0 {
		return c.JSON(fiber.Map{
			"status":  0,
			"message": "No matches found",
		})
	}

	matches := make([]fiber.Map, len(results))
	for i, result := range results {
		matches[i] = fiber.Map{
			"id":        result.ID,
			"file_path": result.FilePath,
		}
	}

	return c.JSON(fiber.Map{
		"status":  1,
		"matches": matches,
	})
}
