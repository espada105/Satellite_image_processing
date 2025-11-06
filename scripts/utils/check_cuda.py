"""
CUDA 설치 및 사용 가능 여부 확인 스크립트
"""

import sys

print("=" * 60)
print("CUDA 설치 및 사용 가능 여부 확인")
print("=" * 60)

# 1. PyTorch 설치 확인
print("\n1️⃣ PyTorch 설치 확인")
try:
    import torch
    print(f"   ✅ PyTorch 설치됨")
    print(f"   버전: {torch.__version__}")
except ImportError:
    print(f"   ❌ PyTorch가 설치되지 않았습니다!")
    print(f"   설치 방법: pip install torch torchvision")
    sys.exit(1)

# 2. CUDA 사용 가능 여부
print("\n2️⃣ CUDA 사용 가능 여부")
cuda_available = torch.cuda.is_available()
print(f"   CUDA 사용 가능: {'✅ 예' if cuda_available else '❌ 아니오'}")

if cuda_available:
    # 3. CUDA 버전 정보
    print("\n3️⃣ CUDA 버전 정보")
    print(f"   PyTorch가 사용하는 CUDA 버전: {torch.version.cuda}")
    print(f"   cuDNN 버전: {torch.backends.cudnn.version()}")
    
    # 4. GPU 정보
    print("\n4️⃣ GPU 정보")
    gpu_count = torch.cuda.device_count()
    print(f"   GPU 개수: {gpu_count}")
    
    for i in range(gpu_count):
        print(f"\n   GPU {i}:")
        print(f"     이름: {torch.cuda.get_device_name(i)}")
        
        props = torch.cuda.get_device_properties(i)
        print(f"     총 메모리: {props.total_memory / 1024**3:.2f} GB")
        print(f"     멀티프로세서 수: {props.multi_processor_count}")
        print(f"     컴퓨팅 능력: {props.major}.{props.minor}")
        
        # 현재 메모리 사용량
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"     현재 할당된 메모리: {allocated:.2f} GB")
        print(f"     현재 예약된 메모리: {reserved:.2f} GB")
        print(f"     사용 가능한 메모리: {(props.total_memory / 1024**3) - reserved:.2f} GB")
    
    # 5. 간단한 테스트
    print("\n5️⃣ CUDA 기능 테스트")
    try:
        # 간단한 텐서 생성 및 연산
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print(f"   ✅ GPU 연산 테스트 성공!")
        print(f"   테스트 결과: {z.shape}")
        
        # 메모리 정리
        del x, y, z
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"   ❌ GPU 연산 테스트 실패: {e}")
else:
    print("\n⚠️  CUDA를 사용할 수 없습니다.")
    print("\n가능한 원인:")
    print("  1. NVIDIA GPU가 설치되지 않음")
    print("  2. CUDA 드라이버가 설치되지 않음")
    print("  3. PyTorch가 CPU 버전으로 설치됨")
    print("  4. GPU 드라이버가 최신 버전이 아님")
    print("\n해결 방법:")
    print("  1. NVIDIA GPU 확인: nvidia-smi 명령어 실행")
    print("  2. CUDA 지원 PyTorch 재설치:")
    print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")

# 6. nvidia-smi 확인 (시스템 명령어)
print("\n6️⃣ 시스템 CUDA 정보 (nvidia-smi)")
try:
    import subprocess
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("   ✅ nvidia-smi 실행 성공")
        # 첫 몇 줄만 출력
        lines = result.stdout.split('\n')[:10]
        for line in lines:
            if line.strip():
                print(f"   {line}")
    else:
        print("   ⚠️  nvidia-smi 실행 실패 (GPU 드라이버가 설치되지 않았을 수 있음)")
except FileNotFoundError:
    print("   ⚠️  nvidia-smi를 찾을 수 없습니다 (GPU 드라이버가 설치되지 않았을 수 있음)")
except subprocess.TimeoutExpired:
    print("   ⚠️  nvidia-smi 실행 시간 초과")
except Exception as e:
    print(f"   ⚠️  nvidia-smi 실행 오류: {e}")

print("\n" + "=" * 60)
print("확인 완료")
print("=" * 60)

