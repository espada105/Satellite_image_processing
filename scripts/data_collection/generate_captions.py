"""
이미지 캡션 생성 스크립트

BLIP-2 모델을 사용하여 전처리된 위성 이미지에 대한 캡션을 자동 생성합니다.

사용법:
    python generate_captions.py --image_dir ./data/processed --output_file ./data/captions/captions.json
    python generate_captions.py --image_dir ./data/processed --output_file ./data/captions/captions.json --batch_size 8 --max_images 100
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
from PIL import Image
import sys
import os

# PyTorch 및 transformers 임포트 (에러 처리)
try:
    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration
    # InstructBLIP (선택적)
    try:
        from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
        HAS_INSTRUCTBLIP = True
    except Exception:
        HAS_INSTRUCTBLIP = False
    # BLIP-2 (선택적)
    try:
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        HAS_BLIP2 = True
    except Exception:
        HAS_BLIP2 = False
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  필요한 라이브러리가 설치되지 않았습니다: {e}")
    print("\n설치 방법:")
    print("  pip install torch torchvision transformers pillow tqdm")
    TRANSFORMERS_AVAILABLE = False
    sys.exit(1)

from tqdm import tqdm

# 프로젝트 루트 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from scripts.utils.config import (
    PROCESSED_DATA_DIR,
    CAPTIONS_DIR
)


def load_caption_model(model_name: str = "Salesforce/blip-image-captioning-large", device: str = None, use_fp16: bool = True):
    """
    BLIP 캡션 생성 모델 로드 (GPU 최적화)
    
    Args:
        model_name: 사용할 모델 이름
        device: 사용할 디바이스 (None이면 자동 선택)
        use_fp16: FP16 사용 여부 (GPU 메모리 절약)
    
    Returns:
        processor, model, device
    """
    print(f"🤖 모델 로딩 중: {model_name}")
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"   디바이스: {device}")
    
    try:
        # 모델 자동 감지: instructblip 지정 시 해당 클래스 사용
        if "instructblip" in model_name.lower():
            if not HAS_INSTRUCTBLIP:
                raise RuntimeError("transformers에 InstructBLIP가 없습니다. pip install -U transformers 필요")
            processor = InstructBlipProcessor.from_pretrained(model_name)
            model = InstructBlipForConditionalGeneration.from_pretrained(model_name)
        elif "blip2" in model_name.lower():
            if not HAS_BLIP2:
                raise RuntimeError("transformers에 BLIP-2가 없습니다. pip install -U transformers 필요")
            processor = Blip2Processor.from_pretrained(model_name)
            # 메모리 최적화: FP16 로딩 시도
            model = Blip2ForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if use_fp16 else None
            )
        else:
            processor = BlipProcessor.from_pretrained(model_name)
            model = BlipForConditionalGeneration.from_pretrained(model_name)
        
        # GPU 최적화
        if device == "cuda":
            model.to(device)
            if use_fp16:
                model.half()  # FP16으로 변환 (메모리 절약)
                print(f"   FP16 모드 활성화 (메모리 절약)")
        else:
            model.to(device)
        
        model.eval()
        
        # GPU 메모리 정보 출력
        if device == "cuda" and torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"   할당된 메모리: {allocated:.2f} GB / 예약된 메모리: {reserved:.2f} GB")
        
        print(f"✅ 모델 로드 완료")
        return processor, model, device
    
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        raise


def create_satellite_prompt(metadata: Dict = None, instruction: str = None) -> str:
    """
    위성 이미지 특화 프롬프트 생성
    
    Args:
        metadata: 메타데이터 딕셔너리 (event_type, location, dataset 등)
    
    Returns:
        프롬프트 문자열
    """
    # 사용자가 제공한 instruction이 있으면 그것을 최상위 지시문으로 사용
    if instruction:
        base_prompt = instruction.strip()
        # instruction이 한 문장 지시형일 수 있으므로, 메타데이터 앵커만 선택적으로 덧붙임
        suffix = []
        if metadata:
            evt = metadata.get('event_type')
            if evt == 'POST-event':
                suffix.append("post-disaster conditions")
            elif evt == 'PRE-event':
                suffix.append("pre-disaster conditions")
            loc = metadata.get('location', {}) if isinstance(metadata, dict) else {}
            loc_name = loc.get('name') if isinstance(loc, dict) else None
            if loc_name:
                suffix.append(f"near {loc_name}")
        if suffix:
            base_prompt += f" (context: {', '.join(suffix)})"
        return base_prompt

    base_prompt = "A detailed satellite image showing"
    
    if metadata:
        # 이벤트 타입 추가
        event_type = metadata.get('event_type', '')
        if event_type == 'POST-event':
            base_prompt += " a post-disaster scene with visible damage and flooding"
        elif event_type == 'PRE-event':
            base_prompt += " a pre-disaster scene with normal conditions"
        
        # 위치 정보 추가
        location = metadata.get('location', {})
        if isinstance(location, dict):
            location_name = location.get('name', '')
            if location_name:
                base_prompt += f" in {location_name}"
    
    # 상세 설명 요청
    base_prompt += ". Describe in detail the terrain, buildings, water bodies, vegetation, roads, and any visible features or changes."
    
    return base_prompt


def generate_captions_batch_gpu(
    image_paths: List[Path],
    processor,
    model,
    device: str,
    max_length: int = 120,
    min_length: int = 20,
    num_beams: int = 5,
    use_fp16: bool = True,
    temperature: float = 0.8,
    repetition_penalty: float = 1.3,
    metadata_dict: Dict[str, Dict] = None,
    use_prompt: bool = True,
    instruction: str = None
) -> List[str]:
    """
    GPU에서 진짜 배치 처리로 여러 이미지의 캡션을 동시에 생성 (품질 개선 버전)
    
    Args:
        image_paths: 이미지 파일 경로 리스트
        processor: BLIP processor
        model: BLIP model
        device: 디바이스
        max_length: 최대 생성 길이 (기본값: 120)
        min_length: 최소 생성 길이 (기본값: 20)
        num_beams: 빔 서치 개수
        use_fp16: FP16 사용 여부 (메모리 절약)
        temperature: 샘플링 온도 (기본값: 0.8)
        repetition_penalty: 반복 방지 페널티 (기본값: 1.3)
        metadata_dict: 이미지 ID를 키로 하는 메타데이터 딕셔너리
        use_prompt: 프롬프트 사용 여부
    
    Returns:
        생성된 캡션 텍스트 리스트
    """
    try:
        # 배치 이미지 로드 및 프롬프트 생성
        images = []
        prompts = []
        valid_paths = []
        
        for image_path in image_paths:
            try:
                image = Image.open(image_path).convert('RGB')
                images.append(image)
                valid_paths.append(image_path)
                
                # 메타데이터 기반 프롬프트 생성
                if use_prompt and metadata_dict:
                    image_id = image_path.stem
                    metadata = metadata_dict.get(image_id, {})
                    prompt = create_satellite_prompt(metadata, instruction=instruction)
                elif use_prompt:
                    prompt = create_satellite_prompt(instruction=instruction)
                else:
                    prompt = None
                
                prompts.append(prompt)
                
            except Exception as e:
                print(f"⚠️  이미지 로드 실패: {image_path.name} - {e}")
                continue
        
        if not images:
            return []
        
        # 배치로 처리 (여러 이미지를 한 번에 GPU에 로드)
        # BLIP-2 모델 감지
        is_blip2 = hasattr(model, '__class__') and 'Blip2' in model.__class__.__name__
        
        if use_prompt and any(p is not None for p in prompts):
            # 프롬프트가 있는 경우 (BLIP/InstructBLIP 모두 지원)
            inputs = processor(images=images, text=prompts, return_tensors="pt", padding=True).to(device)
            
            # BLIP-2의 경우 프롬프트 길이 저장 (나중에 제거하기 위해)
            if is_blip2:
                prompt_input_ids = inputs.get('input_ids', None)
                prompt_lengths = None
                if prompt_input_ids is not None:
                    # 각 프롬프트의 실제 길이 계산 (패딩 제외)
                    prompt_lengths = []
                    for i, prompt in enumerate(prompts):
                        if prompt:
                            # 프롬프트만 토크나이즈하여 길이 계산
                            prompt_tokens = processor.tokenizer(prompt, return_tensors="pt", padding=False)
                            prompt_lengths.append(prompt_tokens['input_ids'].shape[1])
                        else:
                            prompt_lengths.append(0)
        else:
            # 프롬프트가 없는 경우 (기존 방식)
            inputs = processor(images=images, return_tensors="pt").to(device)
            prompt_lengths = None
        
        # 입력도 FP16으로 변환 (모델이 FP16이면)
        if use_fp16 and device == "cuda":
            inputs = {k: v.half() if isinstance(v, torch.Tensor) and v.dtype == torch.float32 else v 
                     for k, v in inputs.items()}
        
        # inputs에서 pad_token_id와 eos_token_id 제거 (모델 내부에서 설정하도록)
        generate_inputs = {k: v for k, v in inputs.items() if k not in ['pad_token_id', 'eos_token_id']}
        
        # generate 파라미터 설정
        # 참고: pad_token_id와 eos_token_id는 BlipForConditionalGeneration.generate()가 
        # 내부적으로 text_decoder.generate()에 전달하므로 여기서 설정하지 않음
        generate_kwargs = {
            "max_length": max_length,
            "min_length": min_length,
            "num_beams": num_beams,
            "do_sample": True,  # ✅ 샘플링 활성화
            "temperature": temperature,  # ✅ 다양성 제어
            "repetition_penalty": repetition_penalty,  # ✅ 반복 방지
            "length_penalty": 1.2,  # ✅ 더 긴 캡션 유도 (1.0보다 크면 긴 시퀀스 선호)
        }
        
        # 배치 생성 (품질 개선 파라미터 적용)
        with torch.no_grad():
            outputs = model.generate(**generate_inputs, **generate_kwargs)
        
        # 배치 결과 디코딩
        captions = processor.batch_decode(outputs, skip_special_tokens=True)
        
        # BLIP-2의 경우 프롬프트 부분 제거
        if is_blip2 and prompt_lengths is not None:
            final_captions = []
            for i, caption in enumerate(captions):
                if prompt_lengths[i] > 0 and prompts[i]:
                    # 프롬프트 부분 제거 (프롬프트가 캡션 시작 부분에 포함된 경우)
                    prompt_text = prompts[i].strip()
                    if caption.startswith(prompt_text):
                        caption = caption[len(prompt_text):].strip()
                    # 또는 프롬프트를 토큰 단위로 제거
                    # 간단하게 프롬프트 텍스트가 포함되어 있으면 제거
                    if prompt_text in caption:
                        # 프롬프트가 캡션의 시작 부분에 있으면 제거
                        parts = caption.split(prompt_text, 1)
                        if len(parts) > 1:
                            caption = parts[1].strip()
                        else:
                            # 프롬프트가 포함되어 있지만 시작 부분이 아니면 그대로 사용
                            pass
                final_captions.append(caption.strip())
            captions = final_captions
        else:
            captions = [caption.strip() for caption in captions]
        
        # 실패한 이미지에 대한 None 처리
        result = []
        valid_idx = 0
        for i, path in enumerate(image_paths):
            if path in valid_paths:
                result.append(captions[valid_idx] if valid_idx < len(captions) else None)
                valid_idx += 1
            else:
                result.append(None)
        
        return result
    
    except Exception as e:
        print(f"⚠️  배치 처리 실패: {e}")
        # 실패 시 빈 리스트 반환
        return [None] * len(image_paths)


def load_metadata_dict(metadata_dir: Path = None) -> Dict[str, Dict]:
    """
    메타데이터를 딕셔너리로 로드 (이미지 ID를 키로 사용)
    
    Args:
        metadata_dir: 메타데이터 디렉토리
    
    Returns:
        {image_id: metadata} 딕셔너리
    """
    if metadata_dir is None:
        from scripts.utils.config import METADATA_DIR
        metadata_dir = METADATA_DIR
    
    if not metadata_dir.exists():
        return {}
    
    metadata_dict = {}
    metadata_files = list(metadata_dir.glob("*_metadata.json"))
    
    for metadata_file in metadata_files:
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata_list = json.load(f)
                for meta in metadata_list:
                    image_id = meta.get('image_id', '')
                    if image_id:
                        metadata_dict[image_id] = meta
        except Exception as e:
            print(f"⚠️  메타데이터 로드 실패: {metadata_file.name} - {e}")
    
    return metadata_dict


def generate_captions_batch(
    image_paths: List[Path],
    processor,
    model,
    device: str,
    batch_size: int = 8,
    max_length: int = 120,
    min_length: int = 20,
    use_fp16: bool = True,
    temperature: float = 0.8,
    repetition_penalty: float = 1.3,
    use_prompt: bool = True,
    metadata_dir: Path = None,
    instruction: str = None
) -> List[Dict]:
    """
    GPU 병렬 처리를 사용한 배치 이미지 캡션 생성 (품질 개선 버전)
    
    Args:
        image_paths: 이미지 파일 경로 리스트
        processor: BLIP processor
        model: BLIP model
        device: 디바이스
        batch_size: 배치 크기 (GPU 메모리에 따라 조정)
        max_length: 최대 생성 길이 (기본값: 120)
        min_length: 최소 생성 길이 (기본값: 20)
        use_fp16: FP16 사용 여부 (메모리 절약)
        temperature: 샘플링 온도 (기본값: 0.8)
        repetition_penalty: 반복 방지 페널티 (기본값: 1.3)
        use_prompt: 프롬프트 사용 여부 (기본값: True)
        metadata_dir: 메타데이터 디렉토리
    
    Returns:
        캡션 정보 리스트
    """
    results = []
    
    # 메타데이터 로드
    metadata_dict = {}
    if use_prompt and metadata_dir:
        metadata_dict = load_metadata_dict(metadata_dir)
        print(f"\n📋 메타데이터 로드: {len(metadata_dict)}개 이미지의 메타데이터 발견")
    
    print(f"\n📝 캡션 생성 시작 (총 {len(image_paths)}개 이미지)")
    print(f"   디바이스: {device}")
    print(f"   배치 크기: {batch_size}")
    print(f"   프롬프트 사용: {use_prompt}")
    if instruction:
        print("   사용자 instruction 적용")
    print(f"   최대 길이: {max_length}, 최소 길이: {min_length}")
    print(f"   Temperature: {temperature}, Repetition Penalty: {repetition_penalty}")
    if device == "cuda":
        print(f"   FP16 사용: {use_fp16}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # GPU 병렬 배치 처리
    for i in tqdm(range(0, len(image_paths), batch_size), desc="캡션 생성"):
        batch_paths = image_paths[i:i + batch_size]
        
        # 진짜 배치 처리 (여러 이미지를 한 번에 GPU에 로드)
        if device == "cuda":
            # GPU 배치 처리 (품질 개선 파라미터 적용)
            captions = generate_captions_batch_gpu(
                batch_paths,
                processor,
                model,
                device,
                max_length=max_length,
                min_length=min_length,
                use_fp16=use_fp16,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                metadata_dict=metadata_dict,
                use_prompt=use_prompt,
                instruction=instruction
            )
        else:
            # CPU는 순차 처리 (메모리 제약, 품질 개선 파라미터 적용)
            captions = []
            for image_path in batch_paths:
                try:
                    image = Image.open(image_path).convert('RGB')
                    
                    # 프롬프트 생성
                    prompt = None
                    if use_prompt:
                        image_id = image_path.stem
                        metadata = metadata_dict.get(image_id, {})
                        prompt = create_satellite_prompt(metadata, instruction=instruction)
                    
                    # 입력 처리
                    if prompt:
                        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
                    else:
                        inputs = processor(images=image, return_tensors="pt").to(device)
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_length=max_length,
                            min_length=min_length,
                            num_beams=5,
                            do_sample=True,
                            temperature=temperature,
                            repetition_penalty=repetition_penalty
                        )
                    caption = processor.decode(outputs[0], skip_special_tokens=True).strip()
                    captions.append(caption)
                except Exception as e:
                    print(f"⚠️  이미지 처리 실패: {image_path.name} - {e}")
                    captions.append(None)
        
        # 결과 정리
        for image_path, caption in zip(batch_paths, captions):
            if caption:
                # 윈도우/절대경로 혼합 환경에서도 안전하게 상대경로 계산
                try:
                    rel_path = image_path.resolve().relative_to(PROCESSED_DATA_DIR.resolve())
                except Exception:
                    rel_path = image_path

                results.append({
                    "image_id": image_path.stem,
                    "image_path": str(rel_path).replace('\\', '/'),
                    "caption": caption
                })
        
        # GPU 메모리 정리 (옵션)
        if device == "cuda" and i % (batch_size * 10) == 0:
            torch.cuda.empty_cache()
    
    return results


def find_all_images(image_dir: Path) -> List[Path]:
    """
    모든 이미지 파일 찾기
    
    Args:
        image_dir: 이미지 디렉토리
    
    Returns:
        이미지 파일 경로 리스트
    """
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(image_dir.rglob(f"*{ext}")))
        image_files.extend(list(image_dir.rglob(f"*{ext.upper()}")))
    
    return sorted(image_files)


def load_metadata_for_captions(captions: List[Dict], metadata_dir: Path) -> List[Dict]:
    """
    메타데이터 정보를 캡션에 추가
    
    Args:
        captions: 캡션 리스트
        metadata_dir: 메타데이터 디렉토리
    
    Returns:
        메타데이터가 추가된 캡션 리스트
    """
    # 메타데이터 파일 로드
    metadata_files = list(metadata_dir.glob("*_metadata.json"))
    metadata_dict = {}
    
    for metadata_file in metadata_files:
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata_list = json.load(f)
                for meta in metadata_list:
                    # 이미지 경로를 기반으로 매칭
                    if 'image_path' in meta:
                        metadata_dict[meta['image_id']] = meta
        except Exception as e:
            print(f"⚠️  메타데이터 로드 실패: {metadata_file.name} - {e}")
    
    # 캡션에 메타데이터 추가
    enriched_captions = []
    for caption_item in captions:
        image_id = caption_item['image_id']
        
        # 메타데이터 찾기 (여러 방법으로 시도)
        metadata = None
        
        # 직접 매칭
        if image_id in metadata_dict:
            metadata = metadata_dict[image_id]
        else:
            # 부분 매칭 시도
            for meta_id, meta in metadata_dict.items():
                if image_id in meta_id or meta_id in image_id:
                    metadata = meta
                    break
        
        enriched_item = caption_item.copy()
        if metadata:
            enriched_item['metadata'] = {
                'location': metadata.get('location', {}),
                'event_type': metadata.get('event_type'),
                'dataset': metadata.get('dataset'),
                'resolution': metadata.get('resolution')
            }
        
        enriched_captions.append(enriched_item)
    
    return enriched_captions


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="이미지 캡션 생성",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 전체 이미지 캡션 생성
  python generate_captions.py --image_dir ./data/processed --output_file ./data/captions/captions.json

  # 테스트용 (100개만)
  python generate_captions.py --image_dir ./data/processed --output_file ./data/captions/captions_test.json --max_images 100

  # GPU 사용, 배치 크기 조정
  python generate_captions.py --image_dir ./data/processed --output_file ./data/captions/captions.json --batch_size 16 --device cuda
        """
    )
    
    parser.add_argument(
        "--image_dir",
        type=str,
        default=str(PROCESSED_DATA_DIR),
        help="전처리된 이미지 디렉토리 (기본값: ./data/processed)"
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        default=str(CAPTIONS_DIR / "captions.json"),
        help="캡션 출력 파일 (기본값: ./data/captions/captions.json)"
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="Salesforce/blip-image-captioning-large",
        help="사용할 모델 이름 (예: Salesforce/blip-image-captioning-large 또는 Salesforce/instructblip-flan-t5-xl)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="배치 크기 (기본값: 8)"
    )
    
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="최대 처리할 이미지 수 (기본값: 전체)"
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=120,
        help="캡션 최대 길이 (기본값: 120)"
    )
    
    parser.add_argument(
        "--min_length",
        type=int,
        default=20,
        help="캡션 최소 길이 (기본값: 20)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="샘플링 온도 (기본값: 0.8, 높을수록 다양함)"
    )
    
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.3,
        help="반복 방지 페널티 (기본값: 1.3, 높을수록 반복 감소)"
    )
    
    parser.add_argument(
        "--no_prompt",
        dest="use_prompt",
        action="store_false",
        help="프롬프트 비활성화 (기본값: 활성화)"
    )
    
    parser.add_argument(
        "--use_prompt",
        dest="use_prompt",
        action="store_true",
        default=True,
        help="프롬프트 사용 (기본값: True)"
    )
    
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="InstructBLIP용 사용자 지시문(영문 권장). 제공 시 메타데이터와 함께 프롬프트로 사용"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help="사용할 디바이스 (기본값: 자동 선택)"
    )
    
    parser.add_argument(
        "--use_fp16",
        action="store_true",
        default=True,
        help="FP16 사용 (GPU 메모리 절약, 기본값: True)"
    )
    
    parser.add_argument(
        "--no_fp16",
        dest="use_fp16",
        action="store_false",
        help="FP16 비활성화"
    )
    
    parser.add_argument(
        "--include_metadata",
        action="store_true",
        help="메타데이터 정보를 캡션에 포함"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("이미지 캡션 생성")
    print("=" * 60)
    
    # 디렉토리 및 파일 경로 설정
    image_dir = Path(args.image_dir)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 이미지 파일 찾기
    print(f"\n🔍 이미지 파일 찾는 중: {image_dir}")
    all_images = find_all_images(image_dir)
    
    if not all_images:
        print("❌ 이미지 파일을 찾을 수 없습니다!")
        sys.exit(1)
    
    print(f"✅ {len(all_images)}개 이미지 파일 발견")
    
    # 최대 이미지 수 제한
    if args.max_images:
        all_images = all_images[:args.max_images]
        print(f"📌 제한 적용: {len(all_images)}개 이미지 처리")
    
    # 모델 로드
    processor, model, device = load_caption_model(
        args.model_name, 
        args.device,
        use_fp16=args.use_fp16
    )
    
    # GPU 배치 크기 자동 조정 (RTX 3060 Ti 8GB 기준)
    if device == "cuda" and args.batch_size == 8:
        # RTX 3060 Ti 8GB에 최적화된 배치 크기 추천
        if args.use_fp16:
            recommended_batch_size = 16  # FP16 사용 시
        else:
            recommended_batch_size = 8   # FP32 사용 시
        
        print(f"\n💡 GPU 배치 크기 추천: {recommended_batch_size} (현재: {args.batch_size})")
        print(f"   메모리 부족 시 --batch_size를 줄이세요")
    
    # 메타데이터 디렉토리 설정
    metadata_dir = None
    if args.use_prompt or args.include_metadata:
        from scripts.utils.config import METADATA_DIR
        metadata_dir = METADATA_DIR
    
    # 캡션 생성 (GPU 병렬 처리, 품질 개선 파라미터 적용)
    captions = generate_captions_batch(
        all_images,
        processor,
        model,
        device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        min_length=args.min_length,
        use_fp16=args.use_fp16,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        use_prompt=args.use_prompt,
        metadata_dir=metadata_dir,
        instruction=args.instruction
    )
    
    print(f"\n✅ 캡션 생성 완료: {len(captions)}개")
    
    # 메타데이터 추가 (선택적)
    if args.include_metadata:
        print("\n📋 메타데이터 추가 중...")
        from scripts.utils.config import METADATA_DIR
        captions = load_metadata_for_captions(captions, METADATA_DIR)
        print("✅ 메타데이터 추가 완료")
    
    # 결과 저장
    print(f"\n💾 캡션 저장 중: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(captions, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 캡션 저장 완료!")
    
    # 통계 출력
    print("\n" + "=" * 60)
    print("생성 완료 통계")
    print("=" * 60)
    print(f"처리된 이미지: {len(captions)}개")
    print(f"저장 위치: {output_file}")
    
    # 샘플 캡션 출력
    if captions:
        print(f"\n📝 샘플 캡션 (처음 3개):")
        for i, item in enumerate(captions[:3], 1):
            print(f"\n{i}. 이미지: {item['image_id']}")
            print(f"   캡션: {item['caption']}")


if __name__ == "__main__":
    main()

