import pandas as pd
import json
import unicodedata
import re
from fuzzywuzzy import fuzz
from tqdm import tqdm
import logging
import os


class AddressHierarchyProcessor:
	def __init__(self, hierarchy_file_path, district_threshold=75, neighborhood_threshold=80):
		"""
		Address hierarchy processor
		- Province: exact match only
		- District: fuzzy matching
		- Neighborhood: fuzzy matching
		"""
		self.district_threshold = district_threshold
		self.neighborhood_threshold = neighborhood_threshold
		self.setup_logging()
		self.hierarchy = self._load_and_normalize_hierarchy(hierarchy_file_path)
		self._create_lookup_tables()
	
	def setup_logging(self):
		"""Configure logging"""
		logging.basicConfig(
			level=logging.INFO,
			format='%(asctime)s - %(levelname)s - %(message)s'
		)
		self.logger = logging.getLogger(__name__)
	
	def _normalize_text(self, text):
		"""Normalize text for consistent comparison"""
		if not text or pd.isna(text):
			return ""
		
		# Unicode normalization
		text = unicodedata.normalize('NFKC', str(text))
		
		# Convert to lowercase
		text = text.lower()
		
		# Handle Turkish-specific characters
		text = text.replace('i̇', 'i')
		
		# Remove extra spaces
		text = re.sub(r'\s+', ' ', text).strip()
		
		return text
	
	def _get_neighborhood_variants(self, neighborhood_name):
		"""
		Generate common variants of a neighborhood name
		"""
		if not neighborhood_name:
			return []
		
		# Normalize
		name = self._normalize_text(neighborhood_name)
		
		# Remove suffixes
		clean_name = re.sub(r'\s*(mahallesi|mahalle|mah)\s*$', '', name).strip()
		
		if not clean_name:
			return []
		
		# Possible variants
		variants = [
			clean_name,
			f"{clean_name} mahalle",
			f"{clean_name} mahallesi",
			f"{clean_name} mah"
		]
		
		return variants
	
	def _load_and_normalize_hierarchy(self, file_path):
		"""Load and normalize the JSON hierarchy file"""
		try:
			with open(file_path, 'r', encoding='utf-8') as f:
				raw_hierarchy = json.load(f)
			
			normalized_hierarchy = {}
			for il, ilceler in raw_hierarchy.items():
				il_normalized = self._normalize_text(il)
				normalized_hierarchy[il_normalized] = {}
				
				for ilce, mahalleler in ilceler.items():
					ilce_normalized = self._normalize_text(ilce)
					# Normalize neighborhood names
					normalized_mahalleler = [
						self._normalize_text(mahalle) for mahalle in mahalleler
					]
					normalized_hierarchy[il_normalized][ilce_normalized] = normalized_mahalleler
			
			self.logger.info(f"Hierarchy loaded: {len(normalized_hierarchy)} provinces")
			return normalized_hierarchy
		
		except Exception as e:
			self.logger.error(f"Error loading hierarchy file: {e}")
			raise
	
	def _create_lookup_tables(self):
		"""Create lookup tables for fast search"""
		self.province_list = list(self.hierarchy.keys())
		self.district_to_province = {}
		self.neighborhood_to_location = {}
		self.neighborhood_variants = {}
		
		# Sort provinces by length (longer first)
		self.province_list.sort(key=len, reverse=True)
		
		for province, districts in self.hierarchy.items():
			for district, neighborhoods in districts.items():
				self.district_to_province[district] = province
				for neighborhood in neighborhoods:
					self.neighborhood_to_location[neighborhood] = (province, district)
					
					# Generate all variants
					variants = self._get_neighborhood_variants(neighborhood)
					for variant in variants:
						self.neighborhood_variants[variant] = neighborhood
		
		self.logger.info(f"Lookup tables created: {len(self.neighborhood_to_location)} neighborhoods")
		self.logger.info(f"Neighborhood variants: {len(self.neighborhood_variants)}")
	
	def _find_province_exact_only(self, address_text):
		"""
		Find province using exact match only
		"""
		address_normalized = self._normalize_text(address_text)
		
		for province in self.province_list:
			pattern = r'\b' + re.escape(province) + r'\b'
			if re.search(pattern, address_normalized):
				return province
		
		return None
	
	def _find_district_with_fuzzy(self, address_text, province):
		"""Find district using exact and fuzzy matching"""
		if not province or province not in self.hierarchy:
			return None
		
		address_normalized = self._normalize_text(address_text)
		districts = list(self.hierarchy[province].keys())
		
		districts.sort(key=len, reverse=True)
		
		# Exact match first
		for district in districts:
			pattern = r'\b' + re.escape(district) + r'\b'
			if re.search(pattern, address_normalized):
				return district
		
		# Fuzzy match if exact not found
		best_match = None
		best_score = 0
		
		words = address_normalized.split()
		for district in districts:
			for word in words:
				if len(word) >= 3:
					score = fuzz.ratio(district, word)
					if score > best_score and score >= self.district_threshold:
						best_score = score
						best_match = district
		
		return best_match
	
	def _find_neighborhood_with_fuzzy(self, address_text, province=None, district=None, debug=False):
		"""Find neighborhood with extended matching logic"""
		address_normalized = self._normalize_text(address_text)
		
		# Limit search to known neighborhoods if province and district are available
		if province and district and province in self.hierarchy and district in self.hierarchy[province]:
			target_neighborhoods = self.hierarchy[province][district]
		else:
			target_neighborhoods = list(self.neighborhood_to_location.keys())
		
		if debug:
			print(f"Debug - Address: {address_normalized}")
			print(f"Debug - Target neighborhoods: {len(target_neighborhoods)}")
		
		# 1. Direct match in variant table
		words = address_normalized.split()
		
		for word in words:
			if word in self.neighborhood_variants:
				original_neighborhood = self.neighborhood_variants[word]
				if original_neighborhood in target_neighborhoods:
					if debug:
						print(f"Debug - Single word match: {word} -> {original_neighborhood}")
					return original_neighborhood
		
		# Multi-word phrases
		for i in range(len(words)):
			for j in range(i + 1, min(i + 4, len(words) + 1)):
				phrase = ' '.join(words[i:j])
				if len(phrase) >= 3:
					if phrase in self.neighborhood_variants:
						original_neighborhood = self.neighborhood_variants[phrase]
						if original_neighborhood in target_neighborhoods:
							if debug:
								print(f"Debug - Phrase match: {phrase} -> {original_neighborhood}")
							return original_neighborhood
		
		# Pattern extraction
		mahalle_patterns = [
			r'\b(\w+(?:\s+\w+)*?)\s+mahalle(?:si)?\b',
			r'\b(\w+(?:\s+\w+)*?)\s+mah\b',
			r'\bmahalle(?:si)?\s+(\w+(?:\s+\w+)*?)\b',
		]
		
		potential_neighborhoods = set()
		
		for pattern in mahalle_patterns:
			matches = re.findall(pattern, address_normalized)
			for match in matches:
				clean_match = match.strip()
				if len(clean_match) >= 2:
					potential_neighborhoods.add(clean_match)
					if debug:
						print(f"Debug - Pattern match: {clean_match}")
		
		for pot_neighborhood in potential_neighborhoods:
			variants = self._get_neighborhood_variants(pot_neighborhood)
			for variant in variants:
				if variant in self.neighborhood_variants:
					original_neighborhood = self.neighborhood_variants[variant]
					if original_neighborhood in target_neighborhoods:
						if debug:
							print(f"Debug - Pattern variant match: {pot_neighborhood} -> {original_neighborhood}")
						return original_neighborhood
		
		# 3. Fuzzy matching
		best_match = None
		best_score = 0
		
		all_candidates = list(potential_neighborhoods) + [w for w in words if len(w) >= 3]
		
		for candidate in all_candidates:
			for neighborhood in target_neighborhoods:
				neighborhood_variants = self._get_neighborhood_variants(neighborhood)
				
				for variant in neighborhood_variants:
					score = fuzz.ratio(candidate, variant)
					if score > best_score and score >= self.neighborhood_threshold:
						best_score = score
						best_match = neighborhood
						if debug:
							print(f"Debug - Fuzzy match: {candidate} ~ {variant} (score: {score})")
		
		return best_match
	
	def _extract_address_components(self, address_text, debug=False):
		"""Extract province, district and neighborhood components"""
		if not address_text or pd.isna(address_text):
			return None, None, None
		
		province = self._find_province_exact_only(address_text)
		if not province:
			return None, None, None
		
		district = self._find_district_with_fuzzy(address_text, province)
		if not district:
			return province, None, None
		
		neighborhood = self._find_neighborhood_with_fuzzy(address_text, province, district, debug)
		
		return province, district, neighborhood
	
	def _create_hierarchical_address(self, province, district, neighborhood, original_address):
		"""Build standardized hierarchical address"""
		if not all([province, district, neighborhood]):
			return original_address
		
		address_normalized = self._normalize_text(original_address)
		
		# Remove matched components from original address
		components_to_remove = [province, district]
		
		neighborhood_variants = self._get_neighborhood_variants(neighborhood)
		components_to_remove.extend(neighborhood_variants)
		
		address_clean = address_normalized
		for component in components_to_remove:
			if component:
				pattern = r'\b' + re.escape(component) + r'\b'
				address_clean = re.sub(pattern, '', address_clean, flags=re.IGNORECASE)
		
		address_clean = re.sub(r'\s+', ' ', address_clean).strip()
		address_clean = re.sub(r'^[,\s\-]+|[,\s\-]+$', '', address_clean)
		
		# Standardize neighborhood format
		neighborhood_clean = re.sub(r'\s*(mahallesi|mahalle|mah)\s*$', '', neighborhood).strip()
		neighborhood_standard = f"{neighborhood_clean} mahalle"
		
		hierarchical_parts = [province, district, neighborhood_standard]
		if address_clean:
			hierarchical_parts.append(address_clean)
		
		return ' '.join(hierarchical_parts)
	
	def process_test_addresses(self, input_csv_path, output_csv_path, debug_count=10):
		"""Process a test CSV file and generate hierarchical addresses"""
		try:
			self.logger.info(f"Reading test CSV file: {input_csv_path}")
			df = pd.read_csv(input_csv_path)
			
			required_columns = ['id', 'address_normalized']
			missing_columns = [col for col in required_columns if col not in df.columns]
			
			if missing_columns:
				self.logger.error(f"Missing required columns: {missing_columns}")
				print("Available columns:", df.columns.tolist())
				return None
			
			df = df.dropna(subset=['address_normalized'])
			total_rows = len(df)
			
			self.logger.info(f"Total addresses to process: {total_rows}")
			
			stats = {
				'processed': 0,
				'hierarchical': 0,
				'province_found': 0,
				'district_found': 0,
				'neighborhood_found': 0
			}
			
			results = []
			
			for idx, row in tqdm(df.iterrows(), total=total_rows, desc="Processing test addresses"):
				test_id = row['id']
				address = row['address_normalized']
				
				debug_mode = idx < debug_count
				
				province, district, neighborhood = self._extract_address_components(address, debug_mode)
				
				stats['processed'] += 1
				if province:
					stats['province_found'] += 1
				if district:
					stats['district_found'] += 1
				if neighborhood:
					stats['neighborhood_found'] += 1
				
				if all([province, district, neighborhood]):
					hierarchical_address = self._create_hierarchical_address(
						province, district, neighborhood, address
					)
					stats['hierarchical'] += 1
				else:
					hierarchical_address = address
				
				results.append({
					'id': test_id,
					'address_sorted': hierarchical_address
				})
				
				if debug_mode:
					print(f"\n{'=' * 50}")
					print(f"Debug - Test ID: {test_id}")
					print(f"Original: {address}")
					print(f"Province: {province}")
					print(f"District: {district}")
					print(f"Neighborhood: {neighborhood}")
					if all([province, district, neighborhood]):
						print(f"✅ Result: {hierarchical_address}")
					else:
						print(f"❌ Missing components - original kept")
					print(f"{'=' * 50}")
			
			result_df = pd.DataFrame(results)
			result_df.to_csv(output_csv_path, index=False, encoding='utf-8')
			
			self._print_statistics(stats, total_rows)
			
			return result_df
		
		except Exception as e:
			self.logger.error(f"Error during test processing: {e}")
			raise
	
	def _print_statistics(self, stats, total):
		"""Print processing statistics"""
		self.logger.info("=" * 60)
		self.logger.info("PROCESSING STATISTICS")
		self.logger.info("=" * 60)
		self.logger.info(f"Total test addresses: {total}")
		self.logger.info(f"Processed: {stats['processed']}")
		self.logger.info(
			f"Province found (exact): {stats['province_found']} ({(stats['province_found'] / total) * 100:.1f}%)")
		self.logger.info(
			f"District found (fuzzy ≥{self.district_threshold}%): {stats['district_found']} ({(stats['district_found'] / total) * 100:.1f}%)")
		self.logger.info(
			f"Neighborhood found (fuzzy ≥{self.neighborhood_threshold}%): {stats['neighborhood_found']} ({(stats['neighborhood_found'] / total) * 100:.1f}%)")
		self.logger.info(f"Fully hierarchical: {stats['hierarchical']} ({(stats['hierarchical'] / total) * 100:.1f}%)")
		self.logger.info("=" * 60)


def main():
	"""Main entry point for test processing"""
	config = {
		'hierarchy_file': "adres_hiyerarsi.json",
		'input_csv': "test_normalized.csv",
		'output_csv': "test_hierarchical.csv",
		'district_threshold': 85,
		'neighborhood_threshold': 75,
		'debug_count': 10
	}
	
	try:
		for file_key in ['hierarchy_file', 'input_csv']:
			if not os.path.exists(config[file_key]):
				print(f"File not found: {config[file_key]}")
				return
		
		print("Starting address hierarchy processing for test file...")
		print("Settings:")
		print(f"   • Input file: {config['input_csv']}")
		print(f"   • Output file: {config['output_csv']}")
		print(f"   • Province: exact match only")
		print(f"   • District: fuzzy matching ≥{config['district_threshold']}%")
		print(f"   • Neighborhood: fuzzy matching ≥{config['neighborhood_threshold']}%")
		print(f"   • Debug: first {config['debug_count']} rows")
		print(f"   • Neighborhood format: standardized 'xxx mahalle'")
		
		processor = AddressHierarchyProcessor(
			hierarchy_file_path=config['hierarchy_file'],
			district_threshold=config['district_threshold'],
			neighborhood_threshold=config['neighborhood_threshold']
		)
		
		result_df = processor.process_test_addresses(
			config['input_csv'],
			config['output_csv'],
			config['debug_count']
		)
		
		if result_df is not None:
			print("\nProcessing completed successfully!")
			print(f"Output file: {config['output_csv']}")
			print(f"Processed records: {len(result_df)}")
			print("All neighborhood names standardized to 'xxx mahalle' format")
			
			print("\nFirst 5 results:")
			print(result_df.head().to_string(index=False))
	
	except Exception as e:
		print(f"Error occurred: {e}")
		logging.exception("Detailed error info:")


if __name__ == "__main__":
	main()
