# Data and model weights: access on request

The VORAFuel dataset and task-specific trained weights are archived in **private
Hugging Face repositories** maintained by Hyobin Park (`hyobin03`). They are not
public downloads. Access is provided on request after author review and any
required source-provider permission.

## How to request access

Open an [issue in this repository](https://github.com/hbp001/VORAFuel/issues) with:

- Your affiliation, if applicable, and intended use.
- The artifacts you need: trained weights, scalers, processed inputs or source data.
- Your Hugging Face username.

Do not post credentials, confidential data or contact details you do not want
public. The maintainer will arrange private follow-up where necessary and confirm
the permitted files, use conditions and delivery method. Requests do not grant
automatic access. Only the reviewed, authorized subset is supplied; the complete
internal reproduction archive is not shared automatically.

## Data provenance and scope

KRISO provided the vessel-operation and fuel measurements. The research team
performed quality control, time alignment, environmental-data integration, model
development and evaluation. Source-provider ownership and restrictions remain in
force. Weather data are not authorized for foundation-model pretraining.

The model archive contains three separate seed checkpoints, scalers and
verification records; it is not an ensemble. The data archive contains the
research reproduction package, including source measurements, processed inputs,
cached representations and reference evaluation artifacts. It does not contain
the complete external weather collection or upstream foundation-model weights.
Checksums accompany the archives.

## 한국어 안내

데이터셋과 학습 가중치는 Hyobin Park의 Hugging Face 비공개 저장소에 보관하며,
저자 검토와 필요한 제공기관의 허가를 거쳐 요청 시 제공합니다.

이 저장소의 Issues에 소속(해당하는 경우), 사용 목적, 필요한 자료·가중치와
Hugging Face 계정을 남겨 주세요. 비밀정보나 공개를 원하지 않는 연락처는
게시하지 마세요. 제공 범위와 사용 조건은 개별 확인 후 안내하며, 내부 재현
패키지 전체를 자동으로 제공하지 않습니다.
