"""
A script for parsing and chunking Word (.docx) files using Docling. The documents are read from a directory (default: `docs`) and the extracted media (images and tables) is stored separately for each document with references to them in the chunk metadata (default: `media/<DOCUMENT_NAME>`). 

Note: Script can also be adapted to process other formats supported by Docling (PDF, HTML, etc.) by changing the file extension filter and ensuring the appropriate Docling converters are installed.

Requires: docling docling-core pandas pillow
Optionally (for tokenizer aware chunking aligned to embedding model): transformers sentence-transformers 
"""

import re
import json

import logging

import pandas as pd

from pathlib import Path
from typing import Any, Optional, Dict, Tuple, List

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter, ConversionResult

from docling_core.types.doc.labels import DocItemLabel
from docling_core.types.doc import PictureItem, TableItem
from docling_core.transforms.chunker.hierarchical_chunker import DocChunk


# Logging
logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
log = logging.getLogger("docx_to_chunks")


# Constants
DOCS: str = "docs"		# source directory
MEDIA: str = "media"	# parent directory to store extracted media from documents
EMBEDDING_MODEL_ID: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2"  # ex: "sentence-transformers/all-MiniLM-L6-v2" for tokenizer-aware chunking
DEFAULT_MAX_TOKENS: int = 512
SUPPORTED_EXTENSIONS: set[str] = {".docx"}


# Helpers

def _sanitise_filename(name: str) -> str:
	"""Replaces problematic characters in file paths."""
	return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(". ")

def _build_chunker(
	embedding_model_id: Optional[str] = None,
	max_tokens: int = DEFAULT_MAX_TOKENS
) -> HybridChunker:
	"""
	Build a HybridChunker, optionally aligned with a HuggingFace tokenizer.
	If `embedding_model_id` is provided (ex: sentence-transfomers/all-MiniLM-L6-v2), 
	the chunker uses that model's tokenizer so that token counts match what the 
	embedding model will see. Otherwise, falls back to the built-in default.
	"""
	if embedding_model_id is not None:
		try:
			from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
			from transformers import AutoTokenizer
			
			hf_tok = AutoTokenizer.from_pretrained(embedding_model_id)
			tokenizer = HuggingFaceTokenizer(tokenizer=hf_tok, max_tokens=max_tokens)
			log.info(f"Using HuggingFace tokenizer from '{embedding_model_id}' with max tokens {max_tokens}.")
			return HybridChunker(tokenizer=tokenizer, merge_peers=True)
		except ImportError as e:
			log.warning(
				f"Failed to initialize HuggingFace tokenizer from '{embedding_model_id}'. Error: {e}"
			)
		
	return HybridChunker(max_tokens=max_tokens, merge_peers=True)


# Media Extraction (images + tables)

def _extract_media(
		doc,
		conv_result,
		media_dir: Path,
		doc_stem: str
) -> Tuple[Dict[str, List[Dict[str, str]]], Dict[str, List[Dict[str, str]]]]:
	"""
	Walk every element in the converted document, and save pictures as PNG and tables as both JSON and CSV.
	
	Returns two look-up dicts keyed by Docling 'self_ref' (ex: "#/pictures/0""):
		- images_map -> { self_ref: [{"path": "...", "format": "png"}] }
		- tables_map -> { self_ref: [{"path": "...", "format": "csv"}, {"path": "...", "format": "json"}] }
	"""
	doc_media_dir = media_dir / _sanitise_filename(doc_stem)
	doc_media_dir.mkdir(parents=True, exist_ok=True)

	images_map: Dict[str, List[Dict[str, str]]] = {}
	tables_map: Dict[str, List[Dict[str, str]]] = {}

	picture_counter = 0
	table_counter = 0

	for element, _level in doc.iterate_items():

		# Pictures
		if isinstance(element, PictureItem):
			picture_counter += 1
			ref = element.self_ref
			saved_paths: List[Dict[str, str]] = []

			# Try to get the PIL image
			# - works reliably for PDF files
			# - for DOCX (Word) files, the image may already be stored as an ImageRef on the element
			try:
				pil_image = element.get_image(doc=doc)
				if pil_image is not None:
					fname = f"picture_{picture_counter}.png"
					fpath = doc_media_dir / fname
					pil_image.save(str(fpath), format="PNG")
					try:
						relative_path = str(fpath.relative_to(Path.cwd()))
					except ValueError:
						relative_path = str(fpath)
					saved_paths.append({"path": relative_path, "format": "png"})
					log.debug(f"Saved picture: {fpath}")
			except Exception as e:
				log.warning(f"Could not export image {ref}: {e}")
			
			# If the element carries a raw image attribute (ex: DOCX)
			if not saved_paths:
				try:
					raw = getattr(element, "image", None)
					if raw is not None:
						pil = getattr(raw, "pil_image", None)
						if pil is not None:
							fname = f"picture_{picture_counter}.png"
							fpath = doc_media_dir / fname
							pil.save(str(fpath), format="PNG")
							try:
								relative_path = str(fpath.relative_to(Path.cwd()))
							except ValueError:
								relative_path = str(fpath)
							saved_paths.append({"path": relative_path, "format": "png"})
							log.debug(f"Saved picture from raw attribute: {fpath}")
				except Exception as e:
					log.warning(f"Could not export image from raw attribute for {ref}: {e}")

			if not saved_paths:
				log.warning(
					f"Picture {ref} has no exportable image data. A placeholder will be used in the chunk metadata, but you may want to check the document and consider converting the image to a supported format."
				)
				
				saved_paths.append({"path": "", "format": "unkwown", "note": "image data unavailable"})
			
			images_map[ref] = saved_paths

		# Tables

		if isinstance(element, TableItem):
			table_counter += 1
			ref = element.self_ref
			saved_paths: List[Dict[str, str]] = []

			try:
				df: pd.DataFrame = element.export_to_dataframe(doc=doc)
				# Saving as CSV for easy viewing
				csv_name = f"table_{table_counter}.csv"
				csv_path = doc_media_dir / csv_name
				df.to_csv(csv_path, index=False)
				try:
					relative_csv_path = str(csv_path.relative_to(Path.cwd()))
				except ValueError:
					relative_csv_path = str(csv_path)
				saved_paths.append({"path": relative_csv_path, "format": "csv"})

				# Saving as JSON (records orientation for row-wise structure)
				json_name = f"table_{table_counter}.json"
				json_path = doc_media_dir / json_name
				df.to_json(json_path, orient="records", indent=2, force_ascii=False)
				try:
					relative_json_path = str(json_path.relative_to(Path.cwd()))
				except ValueError:
					relative_json_path = str(json_path)
				saved_paths.append({"path": relative_json_path, "format": "json"})

				log.debug(f"Saved table: {csv_path.stem} (.csv and .json)")

			except Exception as e:
				log.warning(f"Could not export table {ref}: {e}")
				saved_paths.append({"path": "", "format": "unkwown", "note": str(e)})

	log.info(
		f"Extracted {picture_counter} pictures and {table_counter} tables from document '{doc_stem}'."
	)
	return images_map, tables_map


# Chunking

def _chunks_for_document(
	conv_result: ConversionResult,
	chunker: HybridChunker,
	images_map: Dict[str, List[Dict[str, str]]],
	tables_map: Dict[str, List[Dict[str, str]]],
	source_path: str
) -> List[Dict[str, Any]]:
	"""
	Run the HybridChunker on one converted document and return a list of chunk dictionaries with full metadata, including links to any extracted images or tables that belong to each chunk.
	"""
	doc = conv_result.document
	doc_name = conv_result.input.file.name
	doc_stem = conv_result.input.file.stem

	chunk_iter = chunker.chunk(dl_doc=doc)
	chunks: List[Dict[str, Any]] = []

	for idx, chunk in enumerate(chunk_iter):
		# Enriched text prepends heading context (ideal for embeddings)
		enriched_text = chunker.contextualize(chunk=chunk)
		raw_text = chunk.text

		# Resolve metadata from chunnk.meta
		doc_chunk = DocChunk.model_validate(chunk)
		item_labels: List[str] = []
		item_refs: List[str] = []
		heading_trail: List[str] = []
		chunk_images: List[Dict[str, str]] = []
		chunk_tables: List[Dict[str, str]] = []

		for doc_item in doc_chunk.meta.doc_items:
			label_str = doc_item.label.value if hasattr(doc_item.label, "value") else str(doc_item.label)
			item_labels.append(label_str)
			item_refs.append(doc_item.self_ref)

			# Collect linked media for this chunk
			if doc_item.self_ref in images_map:
				chunk_images.extend(images_map[doc_item.self_ref])
			if doc_item.self_ref in tables_map:
				chunk_tables.extend(tables_map[doc_item.self_ref])

		# Build heading breadcrumb from the chunk's headings metadata
		if hasattr(doc_chunk.meta, "headings") and doc_chunk.meta.headings:
			heading_trail = list(doc_chunk.meta.headings)

		# Determine chunk type
		has_table = any(l in (DocItemLabel.TABLE.value, "table") for l in item_labels)
		has_image = any(l in (DocItemLabel.PICTURE.value, "picture", "image") for l in item_labels)
		text_only = not (has_table or has_image)

		if text_only:
			chunk_type = "text"
		elif has_table and has_image:
			chunk_type = "mixed"
		elif has_table:
			chunk_type = "table"
		elif has_image:
			chunk_type = "image"
		else:
			chunk_type = "unknown"

		# Assemble chunk dictionary
		chunk_id = f"{_sanitise_filename(doc_stem)}_chunk_{idx+1}"

		chunks.append({
			"chunk_id": chunk_id,
			"chunk_index": idx,
			"doc_name": doc_name,
			"source_path": source_path,
			"text": enriched_text,
			"raw_text": raw_text,
			"headings": heading_trail,
			"chunk_type": chunk_type,
			"images": chunk_images,
			"tables": chunk_tables,
			"doc_item_labels": item_labels,
			"doc_item_refs": item_refs,
		})

	log.info(f"Generated {len(chunks)} chunks for document '{doc_stem}'.")
	return chunks


# Runner

def process_directory(
	docx_dir: str | Path,
	media_dir: str | Path = "media",
	embedding_model_id: Optional[str] = None,
	max_tokens: int = DEFAULT_MAX_TOKENS
) -> List[Dict[str, Any]]:
	"""
	Scan 'docx_dir' for .docx files, convert each with Docling, extract media, chunk the content, and return an ordered list of dictionaries ready for vector database ingestion or other downstrea use.

	Parameters:
	- docx_dir: Directory to scan for .docx files.
	- media_dir: Root directory where extracted media (images and tables) are stored, sub-folders are created per document.
	- embedding_model_id: HuggingFace model ID to use for tokenizer-aware chunking (optional).
	- max_tokens: Maximum tokens (or estimated characters) per chunk (default: 512).

	Returns:
	- List of chunk dictionaries with metadata and links to extracted media.
	"""
	docx_dir = Path(docx_dir).resolve()
	media_dir = Path(media_dir).resolve()

	if not docx_dir.is_dir():
		raise FileNotFoundError(f"Directory '{docx_dir}' does not exist or is not a directory.")
	
	# Scan for .docx files (sorted for consistent processing order)
	docx_files = sorted(
		p for p in docx_dir.iterdir()
		if p.suffix.lower() in SUPPORTED_EXTENSIONS and not p.name.startswith("~")
	)

	if not docx_files:
		log.warning(f"No .docx files found in directory '{docx_dir}'.")
		return []
	
	log.info(f"Found {len(docx_files)} .docx files in '{docx_dir}': {[p.name for p in docx_files]}")

	# Initialize converter and chunker
	converter = DocumentConverter()
	chunker = _build_chunker(
		embedding_model_id=embedding_model_id,
		max_tokens=max_tokens
	)

	all_chunks: List[Dict[str, Any]] = []

	for docx_path in docx_files:
		log.info(f"Processing {docx_path.name} ...")

		try:
			conv_result = converter.convert(source=str(docx_path))
		except Exception as e:
			log.error(f"Conversion failed for {docx_path.name}: {e}")
			continue

		doc = conv_result.document
		doc_stem = conv_result.input.file.stem

		# Step 1
		images_map, tables_map = _extract_media(
			doc=doc,
			conv_result=conv_result,
			media_dir=media_dir,
			doc_stem=doc_stem
		)

		# Step 2
		try:
			relative_source_path = str(docx_path.relative_to(Path.cwd()))
		except ValueError:
			relative_source_path = str(docx_path)

		doc_chunks = _chunks_for_document(
			conv_result=conv_result,
			chunker=chunker,
			images_map=images_map,
			tables_map=tables_map,
			source_path=relative_source_path
		)
		all_chunks.extend(doc_chunks)

	log.info(
		f"Processing complete. Total documents processed: {len(docx_files)}. Total chunks generated: {len(all_chunks)}."
	)
	return all_chunks


if __name__ == "__main__":
	chunks = process_directory(
		docx_dir=DOCS,
		media_dir=MEDIA,
		embedding_model_id=EMBEDDING_MODEL_ID,
		max_tokens=DEFAULT_MAX_TOKENS
	)

	# Save chunks to JSON for downstream use
	output_path = Path("chunks_output.json")
	with output_path.open("w", encoding="utf-8") as f:
		json.dump(chunks, f, ensure_ascii=False, indent=2)

	log.info(f"Saved all chunks to {output_path.resolve()}")
