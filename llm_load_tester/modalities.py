from __future__ import annotations

"""
Modality handlers for different input types: Text, Image, and Voice.
"""

import asyncio
import base64
import os
import random
import string
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiofiles


@dataclass
class PayloadResult:
    """Result of payload preparation."""
    payload: dict[str, Any]
    metadata: dict[str, Any]


class ModalityHandler(ABC):
    """Abstract base class for modality handlers."""
    
    @abstractmethod
    async def prepare_payload(self, config: dict[str, Any]) -> PayloadResult:
        """Prepare the API request payload for this modality."""
        pass
    
    @abstractmethod
    def get_presets(self) -> dict[str, Any]:
        """Get preset configurations for this modality."""
        pass

    def resolve_temperature(self, config: dict[str, Any], default: float) -> float:
        """Resolve provider-specific temperature requirements."""
        endpoint = str(config.get("endpoint", ""))
        if "moonshot.ai" in endpoint or "moonshot.cn" in endpoint:
            return 1.0
        return default

    def is_moonshot_endpoint(self, config: dict[str, Any]) -> bool:
        """Check whether the request targets Moonshot."""
        endpoint = str(config.get("endpoint", ""))
        return "moonshot.ai" in endpoint or "moonshot.cn" in endpoint

    def is_modelarts_maas_endpoint(self, config: dict[str, Any]) -> bool:
        """Check whether the request targets Huawei ModelArts MaaS."""
        endpoint = str(config.get("endpoint", ""))
        return "modelarts-maas.com" in endpoint


class DirectoryInputHandler(ModalityHandler):
    """Shared directory-based file selection for multimodal inputs."""

    SUPPORTED_FORMATS: set[str] = set()

    def __init__(self) -> None:
        self._selection_lock: asyncio.Lock | None = None
        self._current_directory: Path | None = None
        self._files: list[Path] = []
        self._next_index = 0

    def _ensure_selection_lock(self) -> asyncio.Lock:
        """Create the selection lock inside a running event loop."""
        if self._selection_lock is None:
            self._selection_lock = asyncio.Lock()
        return self._selection_lock

    def _resolve_path(self, path_str: str) -> Path:
        """Resolve a path string, handling shell escapes, env vars, and home dir."""
        path_str = path_str.replace("\\ ", " ")
        path_str = path_str.replace("\\/", "/")
        path_str = os.path.expanduser(path_str)
        path_str = os.path.expandvars(path_str)
        return Path(path_str)

    def _list_supported_files(self, directory: Path) -> list[Path]:
        """List supported files in a directory."""
        return sorted(
            path for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in self.SUPPORTED_FORMATS
        )

    async def _get_next_file(self, directory: Path, source_label: str) -> Path:
        """Return the next file from a shuffled directory pool."""
        async with self._ensure_selection_lock():
            if self._current_directory != directory:
                self._files = self._list_supported_files(directory)
                if not self._files:
                    raise ValueError(
                        f"No supported {source_label} found in {directory}. "
                        f"Supported: {sorted(self.SUPPORTED_FORMATS)}"
                    )
                random.shuffle(self._files)
                self._current_directory = directory
                self._next_index = 0

            if self._next_index >= len(self._files):
                random.shuffle(self._files)
                self._next_index = 0

            selected_file = self._files[self._next_index]
            self._next_index += 1
            return selected_file


class TextHandler(ModalityHandler):
    """Handler for text-based LLM requests with diverse random prompts."""
    
    TOKEN_PRESETS = [256, 512, 1024, 2048, 4096, 8192]
    
    # Approximate tokens per character for estimation
    CHARS_PER_TOKEN = 4
    
    # Diverse prompt templates by category
    PROMPT_TEMPLATES = {
        "coding": [
            "Write a Python function to {task}. Include error handling, type hints, and docstrings. Explain the time and space complexity.",
            "Implement a {data_structure} in {language}. Include methods for insertion, deletion, and traversal with examples.",
            "Debug this {language} code that {problem}. Identify the issues and provide a corrected version.",
            "Create a {type} API endpoint using {framework}. Include authentication, rate limiting, and request validation.",
            "Design a SQL database schema for {domain}. Include primary keys, foreign keys, indexes, and sample queries.",
            "Write a regular expression to {task}. Explain how it works and provide test cases.",
            "Implement {algorithm} algorithm. Compare it with alternative approaches and analyze performance.",
            "Create a CI/CD pipeline for {platform}. Include build, test, and deployment stages with configuration files.",
        ],
        "writing": [
            "Write a {genre} story about {character} who {situation}. Include dialogue and descriptive setting.",
            "Compose a {type} email to {recipient} regarding {topic}. Use appropriate tone and format.",
            "Draft a technical blog post about {subject}. Include introduction, main points, code examples, and conclusion.",
            "Write a product review for {product}. Include pros, cons, use cases, and recommendation.",
            "Create a {style} poem about {theme}. Use vivid imagery and maintain consistent meter.",
            "Write a persuasive essay arguing {position} about {topic}. Include evidence and counterarguments.",
            "Draft a social media campaign for {brand}. Include post ideas, hashtags, and engagement strategy.",
            "Write a screenplay scene featuring {characters} in {setting}. Include stage directions and dialogue.",
        ],
        "analysis": [
            "Analyze the implications of {trend} on {industry}. Consider short-term and long-term effects.",
            "Compare and contrast {concept_a} versus {concept_b}. Evaluate strengths, weaknesses, and use cases.",
            "Evaluate the ethical considerations of {technology}. Discuss privacy, bias, and societal impact.",
            "Provide a SWOT analysis for {company_or_product}. Include specific examples for each category.",
            "Analyze the historical development of {field}. Identify key milestones and influential figures.",
            "Review the current state of research on {topic}. Summarize findings and identify gaps.",
            "Assess the economic impact of {event} on {region}. Include statistical projections.",
            "Examine the cultural significance of {phenomenon} across different societies and time periods.",
        ],
        "creative": [
            "Design a {type} application that solves {problem}. Include features, user flow, and technical stack.",
            "Create a lesson plan for teaching {subject} to {audience}. Include objectives, activities, and assessment.",
            "Develop a marketing strategy for launching {product} in {market}. Include budget and timeline.",
            "Design a user interface for {purpose}. Consider accessibility, usability, and visual hierarchy.",
            "Create a workout routine for {goal}. Include exercises, sets, reps, and progression plan.",
            "Develop a recipe for {dish} with {dietary_restriction}. Include ingredients, instructions, and nutritional info.",
            "Plan a {duration} itinerary for {destination}. Include activities, transportation, and accommodation.",
            "Design a database migration strategy for {scenario}. Include rollback plan and zero-downtime approach.",
        ],
        " qa": [
            "Explain {concept} as if teaching a {level} student. Use analogies and avoid jargon.",
            "What are the best practices for {activity}? Provide actionable advice with examples.",
            "How does {system} work under the hood? Explain the architecture and data flow.",
            "What are the common pitfalls when {doing_something}? Provide solutions and prevention strategies.",
            "Summarize the key principles of {field}. Include practical applications and case studies.",
            "Describe the process of {procedure} step by step. Include prerequisites and expected outcomes.",
            "What factors should be considered when choosing between {option_a} and {option_b}?",
            "Explain the relationship between {concept_a} and {concept_b} with real-world examples.",
        ],
    }
    
    # Filler variables to populate templates
    FILLER_WORDS = {
        "task": [
            "sort a list of objects by multiple criteria", "parse JSON data from an API response",
            "implement pagination for large datasets", "validate email addresses with domain checking",
            "generate unique identifiers with timestamps", "compress text using run-length encoding",
            "find the longest common subsequence", "calculate moving averages on streaming data",
            "implement a least-recently-used cache", "serialize and deserialize binary trees",
            "detect anagrams in a list of words", "merge overlapping intervals",
            "find all prime numbers up to N using sieve", "implement consistent hashing",
        ],
        "data_structure": [
            "hash map with collision handling", "self-balancing binary search tree",
            "min-max heap with decrease-key operation", "trie for prefix searching",
            "graph using adjacency lists", "circular buffer with thread safety",
            "disjoint set union with path compression", "segment tree for range queries",
            "doubly linked list with sentinel nodes", "bloom filter for membership testing",
        ],
        "language": ["Python", "JavaScript", "Rust", "Go", "Java", "C++", "TypeScript", "Kotlin"],
        "problem": [
            "causes memory leaks in production", "fails to handle concurrent requests",
            "produces incorrect results for edge cases", "runs too slowly for large inputs",
            "crashes when receiving malformed data", "deadlocks under high load",
            "consumes excessive memory", "loses data during network interruptions",
        ],
        "type": ["RESTful", "GraphQL", "WebSocket", "gRPC", "SOAP", "webhook"],
        "framework": ["FastAPI", "Express", "Django", "Spring Boot", "Flask", "Ruby on Rails"],
        "domain": [
            "e-commerce with inventory management", "social media with user relationships",
            "healthcare with patient records", "banking with transaction history",
            "library management with book lending", "food delivery with order tracking",
        ],
        "genre": ["science fiction", "mystery thriller", "historical drama", "romantic comedy", "horror", "fantasy adventure"],
        "character": [
            "a retired detective", "an AI gaining consciousness", "a time traveler",
            "a struggling artist", "a space explorer", "a undercover journalist",
            "a reluctant hero", "a cunning villain", "a curious child",
        ],
        "situation": [
            "discovers a hidden truth", "must save the world in 24 hours",
            "navigates a moral dilemma", "forms an unlikely alliance",
            "uncovers a conspiracy", "faces their greatest fear",
            "makes a life-changing decision", "encounters something inexplicable",
        ],
        "style": ["sonnet", "haiku", "free verse", "narrative", "lyrical", "epic"],
        "theme": ["nature's beauty", "urban isolation", "technological change", "human connection", "loss and memory"],
        "product": [
            "wireless noise-canceling headphones", "smart home security system",
            "portable solar charger", "ergonomic mechanical keyboard",
            "fitness tracking smartwatch", "automated coffee maker",
        ],
        "trend": [
            "remote work adoption", "artificial intelligence integration",
            "sustainable business practices", "decentralized finance",
            "virtual reality commerce", "generative AI content creation",
        ],
        "industry": ["healthcare", "education", "finance", "manufacturing", "retail", "transportation"],
        "concept_a": ["microservices", "monolithic architecture", "SQL databases", "procedural programming"],
        "concept_b": ["serverless functions", "distributed systems", "NoSQL databases", "functional programming"],
        "technology": ["facial recognition", "predictive policing", "algorithmic trading", "deepfake generation"],
        "company_or_product": [
            "a new electric vehicle startup", "a cloud-based CRM platform",
            "an open-source database system", "a subscription meal kit service",
        ],
        "field": ["quantum computing", "biotechnology", "renewable energy", "space exploration"],
        "topic": ["climate change mitigation", "mental health treatment", "autonomous vehicles", "renewable energy storage"],
        "event": ["global pandemic", "major data breach", "trade war", "technological breakthrough"],
        "region": ["Southeast Asia", "European Union", "North America", "Sub-Saharan Africa"],
        "phenomenon": ["social media influence", "urbanization", "globalization", "demographic shifts"],
        "audience": ["elementary students", "college freshmen", "working professionals", "senior citizens"],
        "market": ["emerging markets", "enterprise B2B", " Gen Z consumers", "health-conscious demographics"],
        "purpose": [
            "managing personal finances", "scheduling medical appointments",
            "collaborating on documents", "tracking fitness progress",
            "learning a new language", "monitoring home security",
        ],
        "goal": ["building muscle", "improving cardiovascular health", "increasing flexibility", "weight loss"],
        "dish": ["authentic pasta carbonara", "spicy Thai curry", "vegan chocolate cake", "Mediterranean mezze platter"],
        "dietary_restriction": ["gluten-free ingredients", "keto-friendly macros", "vegan protein sources", "low sodium requirements"],
        "duration": ["weekend", "one-week", "two-week", "month-long"],
        "destination": ["Kyoto Japan", "Iceland's Ring Road", "New Zealand's South Island", "Morocco's Imperial Cities"],
        "scenario": [
            "migrating from MySQL to PostgreSQL", "splitting a monolith into services",
            "adding multi-tenant support", "implementing sharding for scale",
        ],
        "concept": ["blockchain consensus", "neural network backpropagation", "distributed consensus", "quantum entanglement"],
        "level": ["middle school", "high school", "undergraduate", "graduate"],
        "activity": ["code review", "technical interviewing", "incident response", "on-call rotations"],
        "system": ["operating system kernels", "database query optimizers", "web browser rendering", "distributed consensus protocols"],
        "doing_something": ["scaling microservices", "designing APIs", "managing state in React", "optimizing SQL queries"],
        "option_a": ["SQL", "REST", "Python", "AWS"],
        "option_b": ["NoSQL", "GraphQL", "JavaScript", "Azure"],
        "procedure": [
            "deploying to production", "conducting a security audit",
            "performing a database migration", "setting up monitoring and alerting",
        ],
    }
    
    def get_presets(self) -> dict[str, Any]:
        return {
            "input_tokens": self.TOKEN_PRESETS,
            "output_tokens": self.TOKEN_PRESETS
        }
    
    def _get_random_category(self) -> str:
        """Get a random prompt category."""
        return random.choice(list(self.PROMPT_TEMPLATES.keys()))
    
    def _fill_template(self, template: str) -> str:
        """Fill in template variables with random values."""
        result = template
        # Find all {variable} patterns
        import re
        variables = re.findall(r'\{(\w+)\}', template)
        
        for var in variables:
            if var in self.FILLER_WORDS:
                value = random.choice(self.FILLER_WORDS[var])
                result = result.replace(f'{{{var}}}', value, 1)
        
        return result
    
    def _generate_detailed_content(self, base_prompt: str, target_chars: int) -> str:
        """Expand a base prompt with additional details to reach target length."""
        if len(base_prompt) >= target_chars:
            return base_prompt[:target_chars]
        
        # Add context and requirements based on the prompt type
        extensions = [
            "\n\nPlease be thorough and detailed in your response.",
            "\n\nConsider edge cases, best practices, and real-world applications.",
            "\n\nInclude specific examples, code snippets, or references where applicable.",
            "\n\nStructure your response with clear sections and logical flow.",
            "\n\nAddress potential challenges and provide mitigation strategies.",
            "\n\nReference current industry standards and emerging trends.",
            "\n\nExplain your reasoning and decision-making process throughout.",
            "\n\nInclude quantifiable metrics or measurable outcomes where possible.",
        ]
        
        result = base_prompt
        while len(result) < target_chars:
            extension = random.choice(extensions)
            if len(result) + len(extension) <= target_chars:
                result += extension
            else:
                break
        
        # If still need more content, add follow-up questions
        follow_ups = [
            " What are the trade-offs involved?",
            " How would you measure success?",
            " What resources would be required?",
            " What could go wrong and how would you handle it?",
            " Who are the key stakeholders?",
            " What assumptions are you making?",
        ]
        
        while len(result) < target_chars and len(follow_ups) > 0:
            follow_up = random.choice(follow_ups)
            follow_ups.remove(follow_up)
            if len(result) + len(follow_up) <= target_chars:
                result += follow_up
        
        return result[:target_chars]
    
    def generate_random_prompt(self, target_tokens: int) -> str:
        """Generate a diverse, random prompt of approximately target token length."""
        target_chars = target_tokens * self.CHARS_PER_TOKEN
        
        # Select random category and template
        category = self._get_random_category()
        template = random.choice(self.PROMPT_TEMPLATES[category])
        
        # Fill in template variables
        base_prompt = self._fill_template(template)
        
        # Expand to target length
        full_prompt = self._generate_detailed_content(base_prompt, target_chars)
        
        return full_prompt
    
    async def prepare_payload(self, config: dict[str, Any]) -> PayloadResult:
        """Prepare a text-based chat completion payload with random prompt."""
        input_tokens = config.get("input_tokens", 512)
        output_tokens = config.get("output_tokens", 512)
        model = config.get("model", "default-model")

        # Vary the system prompt for most providers, but keep Moonshot text tests
        # deterministic so benchmark runs are comparable and easier to debug.
        system_prompts = [
            "You are a helpful assistant.",
            "You are an expert software engineer with deep knowledge of system design.",
            "You are a creative writer skilled in various genres and styles.",
            "You are a data scientist specializing in machine learning and analytics.",
            "You are a business consultant with expertise in strategy and operations.",
            "You are an educator who explains complex topics clearly.",
            "You are a research analyst who provides evidence-based insights.",
            "You are a technical writer specializing in documentation.",
        ]

        if self.is_moonshot_endpoint(config):
            prompt = (
                "Write a concise technical explanation of how a token bucket rate limiter "
                "works in an API server. Include a simple example and key trade-offs."
            )
            prompt = self._generate_detailed_content(prompt, input_tokens * self.CHARS_PER_TOKEN)
            system_prompt = "You are a precise technical assistant."
        else:
            prompt = self.generate_random_prompt(input_tokens)
            system_prompt = random.choice(system_prompts)
        
        output_token_parameter = config.get("output_token_parameter", "max_tokens")
        if output_token_parameter not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError(
                "output_token_parameter must be 'max_tokens' or 'max_completion_tokens'"
            )

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            output_token_parameter: output_tokens,
            "stream": True,
            "temperature": self.resolve_temperature(
                config,
                random.uniform(0.5, 0.9),
            ),
        }
        thinking_mode = config.get("thinking_mode")
        if thinking_mode in {"enabled", "disabled"}:
            payload["thinking"] = {"type": thinking_mode}
        if self.is_modelarts_maas_endpoint(config):
            payload["stream_options"] = {"include_usage": True}
        
        metadata = {
            "input_tokens": input_tokens,
            "max_output_tokens": output_tokens,
            "output_token_parameter": output_token_parameter,
            "thinking_mode": thinking_mode or "provider_default",
            "prompt_length_chars": len(prompt),
            "modality": "text"
        }
        
        return PayloadResult(payload=payload, metadata=metadata)


class ImageHandler(DirectoryInputHandler):
    """Handler for image-based (multimodal) LLM requests."""
    
    SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
    
    def get_presets(self) -> dict[str, Any]:
        return {"supported_formats": list(self.SUPPORTED_FORMATS)}
    
    def get_mime_type(self, file_path: Path) -> str:
        """Determine MIME type from file extension."""
        ext = file_path.suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp"
        }
        return mime_types.get(ext, "image/jpeg")
    
    async def encode_image(self, image_path: Path) -> str:
        """Encode an image file to base64."""
        async with aiofiles.open(image_path, "rb") as f:
            image_data = await f.read()
        return base64.b64encode(image_data).decode("utf-8")
    
    async def prepare_payload(self, config: dict[str, Any]) -> PayloadResult:
        """Prepare an image-based chat completion payload."""
        image_dir_str = config.get("image_directory", ".")
        image_dir = self._resolve_path(image_dir_str)
        model = config.get("model", "default-model")
        max_tokens = config.get("max_tokens", 512)

        if not image_dir.exists() or not image_dir.is_dir():
            raise ValueError(f"Image directory not found: {image_dir} (resolved from: {image_dir_str})")

        selected_image = await self._get_next_file(image_dir, "images")
        mime_type = self.get_mime_type(selected_image)
        base64_image = await self.encode_image(selected_image)
        
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image in detail."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": max_tokens,
            "stream": True,
            "temperature": self.resolve_temperature(config, 0.7)
        }
        
        metadata = {
            "image_path": str(selected_image),
            "image_size_bytes": selected_image.stat().st_size,
            "mime_type": mime_type,
            "max_output_tokens": max_tokens,
            "modality": "image"
        }
        
        return PayloadResult(payload=payload, metadata=metadata)


class VoiceHandler(DirectoryInputHandler):
    """Handler for voice/audio-based LLM requests."""
    
    SUPPORTED_FORMATS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm"}
    
    def get_presets(self) -> dict[str, Any]:
        return {"supported_formats": list(self.SUPPORTED_FORMATS)}
    
    def get_mime_type(self, file_path: Path) -> str:
        """Determine MIME type from file extension."""
        ext = file_path.suffix.lower()
        mime_types = {
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".m4a": "audio/mp4",
            ".ogg": "audio/ogg",
            ".flac": "audio/flac",
            ".webm": "audio/webm"
        }
        return mime_types.get(ext, "audio/mpeg")
    
    async def encode_audio(self, audio_path: Path) -> str:
        """Encode an audio file to base64."""
        async with aiofiles.open(audio_path, "rb") as f:
            audio_data = await f.read()
        return base64.b64encode(audio_data).decode("utf-8")
    
    async def prepare_payload(self, config: dict[str, Any]) -> PayloadResult:
        """Prepare a voice/audio chat completion payload."""
        audio_directory_str = config.get("audio_directory", config.get("audio_file", "./sound"))
        audio_source = self._resolve_path(audio_directory_str)
        model = config.get("model", "default-model")
        max_tokens = config.get("max_tokens", 512)

        if not audio_source.exists():
            raise ValueError(
                f"Sound path not found: {audio_source} (resolved from: {audio_directory_str})"
            )

        if audio_source.is_dir():
            audio_path = await self._get_next_file(audio_source, "audio files")
            audio_directory = str(audio_source)
        else:
            audio_path = audio_source
            audio_directory = str(audio_source.parent)

        if audio_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported audio format: {audio_path.suffix}. "
                f"Supported: {sorted(self.SUPPORTED_FORMATS)}"
            )

        mime_type = self.get_mime_type(audio_path)
        base64_audio = await self.encode_audio(audio_path)
        
        # Structure follows OpenAI's audio input format
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Transcribe and analyze this audio."
                        },
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": base64_audio,
                                "format": mime_type.replace("audio/", "")
                            }
                        }
                    ]
                }
            ],
            "max_tokens": max_tokens,
            "stream": True,
            "temperature": self.resolve_temperature(config, 0.7)
        }
        
        metadata = {
            "audio_path": str(audio_path),
            "audio_directory": audio_directory,
            "audio_size_bytes": audio_path.stat().st_size,
            "mime_type": mime_type,
            "max_output_tokens": max_tokens,
            "modality": "voice"
        }
        
        return PayloadResult(payload=payload, metadata=metadata)


def get_handler(modality: str) -> ModalityHandler:
    """Factory function to get the appropriate handler for a modality."""
    handlers = {
        "text": TextHandler,
        "image": ImageHandler,
        "voice": VoiceHandler
    }
    
    if modality.lower() not in handlers:
        raise ValueError(f"Unknown modality: {modality}. "
                        f"Supported: {list(handlers.keys())}")
    
    return handlers[modality.lower()]()
