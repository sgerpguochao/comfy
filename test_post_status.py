"""后处理查询路径：静音片优先，未完成时不得回原片。"""
import asyncio
import os
import tempfile
import unittest

import api_server as api


class DummyRequest:
    def __init__(self):
        self.headers = {"host": "example.test:8000"}


class PostStatusTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old_mux = api.MUX_DIR
        api.MUX_DIR = self._tmp.name
        api._muxed.clear()
        api._vo_done.clear()
        api._post_enqueued.clear()
        api._post_failed.clear()
        api._task_noaudio.clear()
        api._task_voiceover.clear()
        api._task_bgm.clear()
        api._video_worker.clear()
        api._post_queue = None

    def tearDown(self):
        api.MUX_DIR = self._old_mux
        self._tmp.cleanup()

    def test_lookup_prefers_muted_on_disk(self):
        tid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        api._task_noaudio.add(tid)
        path = os.path.join(api.MUX_DIR, f"muted_{tid}.mp4")
        with open(path, "wb") as f:
            f.write(b"muted")
        self.assertEqual(api._lookup_processed(tid), f"muted_{tid}.mp4")
        self.assertEqual(api._output_phase(tid)[1], "ready")

    def test_pending_until_mute_exists(self):
        tid = "11111111-2222-3333-4444-555555555555"
        api._task_noaudio.add(tid)
        self.assertEqual(api._output_phase(tid), (None, "pending"))

    def test_skip_when_no_post(self):
        tid = "no-post-needed"
        self.assertEqual(api._output_phase(tid), (None, "skip"))

    def test_resolve_keeps_processing_without_original_url(self):
        tid = "pending-task-id"
        api._task_noaudio.add(tid)
        api._post_queue = asyncio.Queue()
        videos = [{"filename": "minimax_pending_00001_.mp4", "subfolder": "video"}]

        async def _run():
            return await api._resolve_result_videos(tid, 0, videos, DummyRequest())

        phase, extra = asyncio.get_event_loop().run_until_complete(_run())
        self.assertEqual(phase, "processing")
        self.assertTrue(extra.get("audio_processing"))
        self.assertNotIn("videos", extra)
        self.assertEqual(api._post_queue.qsize(), 1)

    def test_resolve_returns_muted_url(self):
        tid = "ready-task-id"
        api._task_noaudio.add(tid)
        muted = f"muted_{tid}.mp4"
        path = os.path.join(api.MUX_DIR, muted)
        with open(path, "wb") as f:
            f.write(b"ok")
        videos = [{"filename": "minimax_ready_00001_.mp4", "subfolder": "video"}]

        async def _run():
            return await api._resolve_result_videos(tid, 0, videos, DummyRequest())

        phase, extra = asyncio.get_event_loop().run_until_complete(_run())
        self.assertEqual(phase, "success")
        self.assertTrue(extra.get("audio_processed"))
        self.assertEqual(extra["videos"][0], f"http://example.test:8000/api/v1/video/{muted}")
        self.assertNotIn("minimax_ready", extra["videos"][0])

    def test_src_missing_allows_retry(self):
        tid = "retry-task"
        api._task_noaudio.add(tid)
        api._post_enqueued.add(tid)

        async def _fake_src(*_a, **_k):
            return None

        orig = api._resolve_src
        api._resolve_src = _fake_src
        try:
            async def _run():
                return await api._post_process(tid, 0, "minimax_x.mp4", "video")

            out = asyncio.get_event_loop().run_until_complete(_run())
            self.assertIsNone(out)
            self.assertNotIn(tid, api._vo_done)
            self.assertNotIn(tid, api._post_enqueued)
        finally:
            api._resolve_src = orig


if __name__ == "__main__":
    unittest.main()
