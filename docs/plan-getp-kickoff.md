# Plan — dựng "khung khởi động" cho candidate (parity với gpt-oss năm ngoái)

## Mục tiêu
Đưa `mla-moe` về đúng trạng thái khởi động mà đề `gpt-oss` năm ngoái trao cho
thí sinh: một engine CPU chạy được + **một file duy nhất được sửa** + **một lệnh
cho ra một con số throughput tok/s trên workload cố định** (điểm perf), trong khi
correctness bị gác bởi harness sẵn có (`make eval`). Bản chất đề giống năm ngoái:
baseline CPU, thí sinh tự port kernel sang GPU.

## Đối chiếu năm ngoái → năm nay

| Năm ngoái (gpt-oss) | Vai trò | Năm nay — hiện trạng | Hành động |
|---|---|---|---|
| `run.cpp` (frozen) chứa `forward()` | engine + oracle tham chiếu | `src/run.c` `forward_unabsorbed`/`forward_absorbed` (đã oracle-validated) | GIỮ frozen, expose prototype qua `include/engine.h` |
| `getp-csrc/getp_eval.cpp` (frozen) | đọc batch, bấm giờ, in `achieved throughput TPS (tok/s)`, ghi output | — CHƯA CÓ | THÊM `src/getp_eval.c` + `include/getp.h` |
| `getp-csrc/getp_run.cpp` (**editable**) `warm_up/finish/inference` | file thí sinh sửa / nơi viết kernel GPU | — CHƯA CÓ | THÊM `src/getp_run.c` (editable surface) |
| `./run model -m getp -i in -o out` | lệnh chấm perf | — CHƯA CÓ | THÊM mode `getp` vào `src/run.c main()` |
| `data/input.txt` (count + lines) | workload cố định | `tests/eval/<model>/requests.txt` (đúng format) | TÁI SỬ DỤNG, không tạo mới |
| `evaluation/` METEOR+BERTScore + threshold | cổng chất lượng | `tests/eval/eval.py` (teacher top-1 + ppl + FUZZY) — giàu hơn | TÁI SỬ DỤNG làm cổng correctness |
| `decode.cpp` | xem output tokens → text | `run -p` + tokenizer đã detokenize | không cần thêm |
| README "Do not modify …" + Quick start | khung đề | README có "Future: candidate hand-off" (chưa chốt) | THÊM mục Quick start + Task + do-not-modify |

## Ranh giới frozen / editable (quyết định thiết kế)
- **Editable (nộp bài): chỉ `src/getp_run.c`.** Thí sinh có thể (a) gọi thẳng
  `forward_*` CPU tham chiếu để chạy đúng ngay, rồi (b) thay dần bằng kernel GPU
  viết trong chính file này. Đây là mô hình y hệt năm ngoái: forward CPU đông
  cứng là oracle, thí sinh viết đường đi nhanh trong file của mình.
- **Frozen:** `run.c`, `model_load.c`, `safetensors_loader.c`, `tokenizer.c`,
  `dump.c`, `main.c`, `getp_eval.c`, toàn bộ `tests/`, và `include/*`.
- **Điểm cần bạn xác nhận (judgment call):** với port GPU thí sinh có thể cần
  đổi trình biên dịch/flags trong `Makefile` (năm ngoái Makefile frozen nhưng đã
  mặc định `hipcc --offload-arch=gfx90a`; năm nay mặc định `clang`). Tôi để
  `Makefile` KHÔNG nằm trong danh sách frozen cứng và ghi chú rõ, thay vì tự
  quyết đóng băng nó.

## Cách đo throughput (frozen harness)
`getp()` trong `getp_eval.c`:
1. đọc `requests.txt` (dòng 0 = số request, N dòng prompt text);
2. `warm_up(t)` — bấm giờ riêng (không tính vào throughput);
3. bấm giờ quanh `inference(t, &reqs)` → tổng token sinh ra / thời gian →
   in `achieved throughput TPS (tok/s)`;
4. ghi file output: mỗi dòng là dãy token id sinh ra (space-separated, giống
   `completions.i32.txt`) để có thể chấm chất lượng lại;
5. `finish(t)` — bấm giờ riêng.

## Hợp đồng file editable (`getp_run.c`)
```c
void      warm_up(Transformer *t);              // cấp phát, upload weight lên GPU…
void      finish(Transformer *t);               // giải phóng
long long inference(Transformer *t, Requests *reqs);  // trả tổng token sinh ra
```
Bản tham chiếu: mỗi request → tokenize (add_bos=0) → `forward_unabsorbed` prefill
→ greedy `forward_absorbed` tới EOS hoặc `max_steps` → lưu token vào `reqs`.

## Việc cần làm
1. `include/engine.h` — prototype `forward_unabsorbed/forward_absorbed/sample`.
2. `include/getp.h` — `Requests` + API `getp/warm_up/finish/inference`.
3. `src/getp_eval.c` (frozen) — harness đọc/bấm giờ/ghi/in throughput.
4. `src/getp_run.c` (editable) — bản tham chiếu CPU.
5. `src/run.c` — thêm dispatch mode `getp` trong `main()` (thay đổi tối thiểu).
6. `Makefile` — thêm 2 src vào `run`, thêm target `make getp MODEL=…`.
7. `README.md` — mục Quick start / Task / do-not-modify / cách chấm.

## Kiểm chứng
- `make` build sạch (đã có 2 file mới).
- `./run "$DSV" getp tests/eval/dsv2lite/requests.txt /tmp/out.txt` in ra
  `achieved throughput TPS (tok/s): …` và ghi `/tmp/out.txt`.
- `make eval MODEL=dsv2lite` vẫn pass (không đụng đường correctness).
