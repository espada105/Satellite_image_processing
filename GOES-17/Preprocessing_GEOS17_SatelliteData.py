import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import os
import glob
from scipy.ndimage import gaussian_filter
import cv2
from datetime import datetime

def extract_satellite_images(data_dir=None, output_dir=None, file_pattern="baseband_*.wav"):
    """
    위성 신호 파일에서 이미지를 추출하는 주 함수
    
    Parameters:
    data_dir (str): 원시 데이터가 있는 디렉토리 경로
    output_dir (str): 결과물을 저장할 디렉토리 경로
    file_pattern (str): 처리할 파일 패턴
    """
    # 1. 데이터 폴더 및 파일 경로 설정
    if data_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, "Data", "rawdata")
    
    if output_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(current_dir, "Output")
    
    print(f"데이터 디렉토리: {data_dir}")
    print(f"출력 디렉토리: {output_dir}")
    
    # 출력 디렉토리가 없으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 2. baseband 파일 목록 가져오기
    baseband_files = glob.glob(os.path.join(data_dir, file_pattern))
    print(f"총 {len(baseband_files)}개의 baseband 파일을 찾았습니다.")
    
    if len(baseband_files) == 0:
        print("파일을 찾지 못했습니다. 경로를 확인해주세요.")
        return
    
    for file_idx, file_path in enumerate(baseband_files):
        file_name = os.path.basename(file_path)
        print(f"[{file_idx+1}/{len(baseband_files)}] 처리 중: {file_name}")
        
        try:
            # 파일 처리 및 이미지 생성
            process_satellite_file(file_path, output_dir)
        except Exception as e:
            print(f"오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("모든 파일 처리 완료!")

# process_satellite_file 함수 수정
def process_satellite_file(file_path, output_dir):
    """
    Parameters:
    file_path (str): 처리할 파일 경로
    output_dir (str): 결과물을 저장할 디렉토리 경로
    """
    file_name = os.path.basename(file_path)
    file_base = os.path.splitext(file_name)[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # 파일 로드
        print("파일 로딩 중...")
        sample_rate, data = wavfile.read(file_path)
        print(f"샘플링 레이트: {sample_rate} Hz")
        print(f"데이터 형태: {data.shape}")
        
        # 모노 채널 확보
        if len(data.shape) > 1:
            print("스테레오에서 모노로 변환 중...")
            data = data[:, 0]
        
        # 데이터 크기 제한 (메모리 문제 방지를 위해 더 작게 설정)
        max_samples = min(1000000, len(data))  # 최대 1백만 샘플로 제한
        print(f"데이터 제한: {len(data)} -> {max_samples} 샘플")
        data = data[:max_samples]
        
        # 정규화
        print("데이터 정규화 중...")
        data = data.astype(np.float32) / np.max(np.abs(data))
        
        # 데이터 잡음 제거
        print("신호 잡음 제거 중...")
        data = denoise_signal(data)
        
        # 스펙트로그램 이미지 생성
        print("스펙트로그램 이미지 생성 중...")
        extract_spectrogram_image(data, sample_rate, file_base, output_dir, timestamp)
        
        # 엔벨로프 이미지 생성 (문제 지점일 수 있음)
        print("엔벨로프 이미지 생성 중...")
        # 타임아웃 메커니즘 추가
        try:
            extract_envelope_image(data, sample_rate, file_base, output_dir, timestamp)
            print("엔벨로프 이미지 생성 완료")
        except Exception as e:
            print(f"엔벨로프 이미지 생성 중 오류: {str(e)}")
        
        # 동기화 기반 이미지 생성 (또 다른 문제 지점일 수 있음)
        print("동기화 기반 이미지 생성 중...")
        try:
            extract_sync_based_image(data, sample_rate, file_base, output_dir, timestamp)
            print("동기화 기반 이미지 생성 완료")
        except Exception as e:
            print(f"동기화 기반 이미지 생성 중 오류: {str(e)}")
        
        print(f"{file_name} 처리 완료!")
        
    except Exception as e:
        print(f"파일 처리 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

def denoise_signal(data, filter_order=3, cutoff_freq=0.05):
    """
    신호 잡음 제거 및 필터링
    
    Parameters:
    data (numpy.ndarray): 입력 신호 데이터
    filter_order (int): 필터 차수
    cutoff_freq (float): 차단 주파수 (Nyquist 주파수 대비 비율)
    
    Returns:
    numpy.ndarray: 필터링된 신호
    """
    # 메디안 필터로 스파이크 노이즈 제거
    data_median_filtered = signal.medfilt(data, kernel_size=5)
    
    # 저역 통과 필터로 고주파 노이즈 제거
    b, a = signal.butter(filter_order, cutoff_freq, 'low')
    data_filtered = signal.filtfilt(b, a, data_median_filtered)
    
    return data_filtered

def extract_spectrogram_image(data, sample_rate, file_base, output_dir, timestamp):
    """
    스펙트로그램 기반 이미지 생성
    
    Parameters:
    data (numpy.ndarray): 입력 신호 데이터
    sample_rate (int): 샘플링 레이트
    file_base (str): 기본 파일 이름
    output_dir (str): 출력 디렉토리
    timestamp (str): 타임스탬프
    """
    plt.figure(figsize=(12, 10))
    
    # 스펙트로그램 계산 - 해상도를 높이기 위해 nperseg 값 조정
    f, t, Sxx = signal.spectrogram(data, sample_rate, nperseg=4096, noverlap=2048)
    
    # 스펙트로그램 강화
    Sxx_log = 10 * np.log10(Sxx + 1e-10)
    
    # 다이나믹 레인지 조정 (상위/하위 백분위수로 클리핑)
    vmin = np.percentile(Sxx_log, 5)
    vmax = np.percentile(Sxx_log, 95)
    
    # 시각화 - 컬러맵을 viridis로 설정하여 지구 이미지처럼 보이도록
    plt.pcolormesh(t, f, Sxx_log, cmap='viridis', vmin=vmin, vmax=vmax)
    plt.ylabel('주파수 (Hz)')
    plt.xlabel('시간 (초)')
    plt.title(f'위성 이미지 재구성 (스펙트로그램 기반)')
    plt.colorbar(label='신호 강도 (dB)')
    
    # 이미지 경계 제거
    plt.axis('off')
    plt.tight_layout(pad=0)
    
    # 이미지 저장
    spectrogram_image_path = os.path.join(output_dir, f"{file_base}_spectrogram_{timestamp}.png")
    plt.savefig(spectrogram_image_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print(f"스펙트로그램 이미지가 생성되었습니다: {spectrogram_image_path}")
    
    # 피클 데이터 저장 (후속 처리를 위해)
    np.savez(os.path.join(output_dir, f"{file_base}_spectrogram_data_{timestamp}.npz"), 
             f=f, t=t, Sxx=Sxx, Sxx_log=Sxx_log)

def extract_envelope_image(data, sample_rate, file_base, output_dir, timestamp):
    """
    신호 엔벨로프 기반 이미지 생성
    
    Parameters:
    data (numpy.ndarray): 입력 신호 데이터
    sample_rate (int): 샘플링 레이트
    file_base (str): 기본 파일 이름
    output_dir (str): 출력 디렉토리
    timestamp (str): 타임스탬프
    """
    plt.figure(figsize=(12, 10))
    
    # 힐버트 변환으로 신호 엔벨로프 추출
    analytic_signal = signal.hilbert(data)
    amplitude_envelope = np.abs(analytic_signal)
    
    # 위상 정보 추출 (일부 위성 이미지에서 유용할 수 있음)
    phase_info = np.angle(analytic_signal)
    
    # 낮은 차단 주파수로 필터링하여 느린 변화 신호만 추출
    b, a = signal.butter(3, 0.01)
    filtered_envelope = signal.filtfilt(b, a, amplitude_envelope)
    
    # 가우시안 필터로 평활화
    filtered_envelope = gaussian_filter(filtered_envelope, sigma=2)
    
    # 이미지로 재구성 (2D로 재구성)
    # GOES-17 이미지 라인을 형성하기 위한 추정값
    # (실제 값은 신호 세부 사항에 따라 다를 수 있음)
    line_samples = estimate_line_length(filtered_envelope, sample_rate)
    rows = len(filtered_envelope) // line_samples
    
    # 2D 이미지로 재구성
    image_data = np.zeros((rows, line_samples))
    for i in range(rows):
        start = i * line_samples
        end = start + line_samples
        if end <= len(filtered_envelope):
            image_data[i, :] = filtered_envelope[start:end]
    
    # 이미지 정규화 및 시각화
    image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))
    
    # 이미지 대비 향상
    image_data = enhance_image_contrast(image_data)
    
    plt.imshow(image_data, cmap='inferno', aspect='auto')
    plt.title('위성 신호 재구성 이미지 (엔벨로프 기반)')
    plt.axis('off')
    
    # 이미지 저장
    envelope_image_path = os.path.join(output_dir, f"{file_base}_envelope_{timestamp}.png")
    plt.savefig(envelope_image_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print(f"엔벨로프 기반 이미지가 생성되었습니다: {envelope_image_path}")
    
    # 위상 정보 이미지 저장
    plt.figure(figsize=(12, 10))
    
    # 위상 데이터 2D 변환
    phase_image = np.zeros((rows, line_samples))
    for i in range(rows):
        start = i * line_samples
        end = start + line_samples
        if end <= len(phase_info):
            phase_image[i, :] = phase_info[start:end]
    
    plt.imshow(phase_image, cmap='twilight', aspect='auto')
    plt.title('위성 신호 위상 정보 이미지')
    plt.axis('off')
    
    phase_image_path = os.path.join(output_dir, f"{file_base}_phase_{timestamp}.png")
    plt.savefig(phase_image_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print(f"위상 정보 이미지가 생성되었습니다: {phase_image_path}")
    
    # NumPy 파일로 저장 (후속 처리를 위해)
    np.save(os.path.join(output_dir, f"{file_base}_envelope_data_{timestamp}.npy"), image_data)

def extract_sync_based_image(data, sample_rate, file_base, output_dir, timestamp):
    """
    동기화 패턴 기반 이미지 생성
    
    Parameters:
    data (numpy.ndarray): 입력 신호 데이터
    sample_rate (int): 샘플링 레이트
    file_base (str): 기본 파일 이름
    output_dir (str): 출력 디렉토리
    timestamp (str): 타임스탬프
    """
    # 동기화 패턴 검출
    sync_positions = detect_sync_patterns(data)
    
    if len(sync_positions) < 2:
        print("동기화 패턴을 충분히 찾지 못했습니다. 기본 방법으로 대체합니다.")
        # 기본 라인 길이 계산
        line_samples = estimate_line_length(data, sample_rate)
    else:
        # 동기화 패턴 간 평균 거리 계산
        sync_diffs = np.diff(sync_positions)
        line_samples = int(np.median(sync_diffs))  # 중앙값 사용으로 이상치 영향 감소
        print(f"검출된 평균 스캔 라인 길이: {line_samples} 샘플")
    
    # 동기화 패턴으로 이미지 구성
    plt.figure(figsize=(12, 10))
    
    # 힐버트 변환으로 신호 엔벨로프 추출
    analytic_signal = signal.hilbert(data)
    amplitude_envelope = np.abs(analytic_signal)
    
    # 저역 통과 필터링
    b, a = signal.butter(3, 0.01)
    filtered_envelope = signal.filtfilt(b, a, amplitude_envelope)
    
    # 스캔 라인 기반으로 이미지 재구성
    rows = len(filtered_envelope) // line_samples
    image_data = np.zeros((rows, line_samples))
    
    # 이미지 채우기 - 동기화 패턴 기반
    if len(sync_positions) >= rows:
        for i in range(rows-1):
            start = sync_positions[i]
            length = min(line_samples, len(filtered_envelope) - start)
            if start + length <= len(filtered_envelope):
                image_data[i, :length] = filtered_envelope[start:start+length]
    else:
        # 충분한 동기화 패턴이 없으면 기본 방법으로
        for i in range(rows):
            start = i * line_samples
            end = start + line_samples
            if end <= len(filtered_envelope):
                image_data[i, :] = filtered_envelope[start:end]
    
    # 이미지 정규화 및 대비 향상
    image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))
    image_data = enhance_image_contrast(image_data)
    
    # 잡음 제거 및 선명화
    image_data = cv2.GaussianBlur(image_data, (3, 3), 0)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    image_data = cv2.filter2D(image_data, -1, kernel)
    
    plt.imshow(image_data, cmap='gist_earth', aspect='auto')
    plt.title('위성 신호 재구성 이미지 (동기화 패턴 기반)')
    plt.axis('off')
    
    # 이미지 저장
    sync_image_path = os.path.join(output_dir, f"{file_base}_sync_based_{timestamp}.png")
    plt.savefig(sync_image_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print(f"동기화 패턴 기반 이미지가 생성되었습니다: {sync_image_path}")
    
    # NumPy 파일로 저장 (후속 처리를 위해)
    np.save(os.path.join(output_dir, f"{file_base}_sync_data_{timestamp}.npy"), image_data)
    
    # 합성 컬러 이미지 시도 (의사 컬러)
    create_false_color_image(image_data, file_base, output_dir, timestamp)

def detect_sync_patterns(data, threshold_factor=0.7, window_size=1000):
    """
    신호에서 동기화 패턴 위치 검출
    
    Parameters:
    data (numpy.ndarray): 입력 신호 데이터
    threshold_factor (float): 임계값 계수
    window_size (int): 검색 윈도우 크기
    
    Returns:
    numpy.ndarray: 동기화 패턴 위치 인덱스
    """
    # 신호 절대값
    abs_data = np.abs(data)
    
    # 이동 평균 계산
    window = np.ones(window_size) / window_size
    moving_avg = np.convolve(abs_data, window, mode='same')
    
    # 진폭 임계값 계산 (이동 평균의 일정 비율 이상)
    threshold = threshold_factor * np.mean(moving_avg)
    
    # 임계값보다 높은 피크 찾기
    peaks, _ = signal.find_peaks(abs_data, height=threshold, distance=window_size//2)
    
    return peaks

def estimate_line_length(data, sample_rate):
    """
    위성 신호의 스캔 라인 길이를 추정
    
    Parameters:
    data (numpy.ndarray): 입력 신호 데이터
    sample_rate (int): 샘플링 레이트
    
    Returns:
    int: 추정된 라인 길이 (샘플 수)
    """
    # 자기상관 계산으로 주기성 찾기
    correlation = np.correlate(data, data, mode='full')
    correlation = correlation[len(correlation)//2:]  # 양의 지연만 고려
    
    # 첫 번째 피크 이후의 다음 피크 찾기 (자기 자신 제외)
    peaks, _ = signal.find_peaks(correlation, height=0.1*np.max(correlation), distance=sample_rate//10)
    
    if len(peaks) > 1:
        line_length = peaks[1]  # 두 번째 피크 위치
    else:
        # 피크를 찾지 못하면 기본값 사용 (0.5초당 라인)
        line_length = int(sample_rate * 0.5)
    
    # 범위 제한 (비현실적인 값 방지)
    min_length = int(sample_rate * 0.1)  # 최소 0.1초
    max_length = int(sample_rate * 2.0)  # 최대 2초
    
    if line_length < min_length:
        line_length = min_length
    elif line_length > max_length:
        line_length = max_length
    
    return line_length

def enhance_image_contrast(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    CLAHE 알고리즘을 사용하여 이미지 대비 향상
    
    Parameters:
    image (numpy.ndarray): 입력 이미지
    clip_limit (float): 히스토그램 클리핑 제한
    tile_grid_size (tuple): 타일 그리드 크기
    
    Returns:
    numpy.ndarray: 대비가 향상된 이미지
    """
    # 이미지 값 범위를 0-255로 변환
    img_uint8 = (image * 255).astype(np.uint8)
    
    # CLAHE 적용
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(img_uint8)
    
    # 다시 0-1 범위로 정규화
    return enhanced.astype(np.float32) / 255.0

def create_false_color_image(image_data, file_base, output_dir, timestamp):
    """
    의사 컬러 위성 이미지 생성 (RGB 채널 합성)
    
    Parameters:
    image_data (numpy.ndarray): 입력 이미지 데이터
    file_base (str): 기본 파일 이름
    output_dir (str): 출력 디렉토리
    timestamp (str): 타임스탬프
    """
    plt.figure(figsize=(12, 10))
    
    # 원본 이미지 데이터를 기반으로 3개의 약간 다른 채널 생성
    # 이 방법은 서로 다른 주파수 대역을 모방합니다
    
    # 적색 채널 - 원본 데이터 약간 블러
    red_channel = gaussian_filter(image_data, sigma=1.5)
    
    # 녹색 채널 - 원본 데이터
    green_channel = image_data.copy()
    
    # 청색 채널 - 원본 데이터 미분 (엣지 강조)
    blue_channel = np.diff(image_data, axis=1, prepend=0)
    blue_channel = (blue_channel - np.min(blue_channel)) / (np.max(blue_channel) - np.min(blue_channel))
    
    # RGB 이미지 생성
    rgb_image = np.zeros((image_data.shape[0], image_data.shape[1], 3))
    rgb_image[:,:,0] = red_channel
    rgb_image[:,:,1] = green_channel
    rgb_image[:,:,2] = blue_channel
    
    # 클리핑
    rgb_image = np.clip(rgb_image, 0, 1)
    
    plt.imshow(rgb_image)
    plt.title('위성 이미지 의사 컬러 합성')
    plt.axis('off')
    
    # 이미지 저장
    false_color_path = os.path.join(output_dir, f"{file_base}_false_color_{timestamp}.png")
    plt.savefig(false_color_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print(f"의사 컬러 이미지가 생성되었습니다: {false_color_path}")

def create_colormap_variants(output_dir, file_base, timestamp):
    """
    다양한 컬러맵으로 이미지 변형 생성
    
    Parameters:
    output_dir (str): 출력 디렉토리
    file_base (str): 기본 파일 이름
    timestamp (str): 타임스탬프
    """
    # 데이터 불러오기 (가장 최근에 생성된 이미지 데이터)
    try:
        image_path = os.path.join(output_dir, f"{file_base}_sync_data_{timestamp}.npy")
        if not os.path.exists(image_path):
            image_path = os.path.join(output_dir, f"{file_base}_envelope_data_{timestamp}.npy")
        
        if os.path.exists(image_path):
            image_data = np.load(image_path)
            
            # 다양한 컬러맵으로 생성
            colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis',
                         'gist_earth', 'gist_heat', 'nipy_spectral', 'jet']
            
            for cmap_name in colormaps:
                plt.figure(figsize=(12, 10))
                plt.imshow(image_data, cmap=cmap_name, aspect='auto')
                plt.title(f'위성 이미지 ({cmap_name} 컬러맵)')
                plt.axis('off')
                
                # 이미지 저장
                cmap_image_path = os.path.join(output_dir, f"{file_base}_{cmap_name}_{timestamp}.png")
                plt.savefig(cmap_image_path, dpi=300, bbox_inches='tight', pad_inches=0)
                plt.close()
            
            print(f"다양한 컬러맵 변형 이미지가 생성되었습니다.")
    except Exception as e:
        print(f"컬러맵 변형 생성 중 오류: {str(e)}")

def post_process_images(output_dir):
    """
    생성된 이미지 후처리 (지형 보정, 투영 등)
    
    Parameters:
    output_dir (str): 이미지가 저장된 디렉토리
    """
    # 이미지 파일 목록 가져오기
    image_files = glob.glob(os.path.join(output_dir, "*.png"))
    
    for image_file in image_files:
        try:
            # 이미지 로드
            img = cv2.imread(image_file)
            
            if img is None:
                continue
                
            # 노이즈 제거
            denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
            
            # 이미지 선명화
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(denoised, -1, kernel)
            
            # 저장 (원본 이름에 "_enhanced" 추가)
            file_name = os.path.basename(image_file)
            base_name, ext = os.path.splitext(file_name)
            enhanced_path = os.path.join(output_dir, f"{base_name}_enhanced{ext}")
            cv2.imwrite(enhanced_path, sharpened)
            
        except Exception as e:
            print(f"이미지 후처리 중 오류: {str(e)}")

# 메인 함수
if __name__ == "__main__":
    # 데이터 경로 설정 (필요한 경우 변경)
    data_directory = None  # 기본 경로 사용
    output_directory = None  # 기본 경로 사용
    
    # 위성 이미지 추출 실행
    extract_satellite_images(data_directory, output_directory)
    
    # 추출된 이미지 후처리 (선택 사항)
    post_process_images(output_directory)