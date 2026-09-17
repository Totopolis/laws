"""Дым-тест MCP-сервера: поднять как подпроцесс и вызвать инструменты по протоколу.

    python -m laws_mcp.smoke
"""
import asyncio, json, os, sys, time

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    env = dict(os.environ, LAWS_BACKEND="onnx",
               PYTHONPATH=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
               PYTHONIOENCODING="utf-8")
    params = StdioServerParameters(command=sys.executable, args=["-m", "laws_mcp.server"], env=env)
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = await s.list_tools()
            print("инструменты:", [t.name for t in tools.tools])

            t0 = time.time()
            res = await s.call_tool("search_law", {"query": "что грозит за неуплату алиментов", "top_k": 3})
            print(f"\nsearch_law ({time.time()-t0:.1f} с, первый вызов — с загрузкой модели):")
            for h in json.loads(res.content[0].text):
                print(f"  {h['code_slug']:<16} {h['article'][:60]:<62} {h['score']}")

            t0 = time.time()
            res = await s.call_tool("search_law", {"query": "самовольная постройка снос", "top_k": 3,
                                                   "codes": ["gk-rf-chast-1"]})
            print(f"\nsearch_law с фильтром по кодексу ({time.time()-t0:.2f} с):")
            for h in json.loads(res.content[0].text):
                print(f"  {h['code_slug']:<16} {h['article'][:60]}")

            res = await s.call_tool("get_article", {"code_slug": "uk-rf", "number": "105"})
            art = json.loads(res.content[0].text)
            print(f"\nget_article uk-rf 105: {art['heading']}, частей: {len(art['content'])}")

            res = await s.call_tool("get_article", {"code_slug": "bk-rf", "number": "242.1"})
            got = json.loads(res.content[0].text)
            print(f"get_article bk-rf 242.1 (номер-дубль): вернулось {len(got) if isinstance(got, list) else 1} шт.")

            res = await s.call_tool("list_codes", {})
            reg = json.loads(res.content[0].text)
            print(f"\nlist_codes: снапшот {reg['snapshot']}, кодексов {len(reg['codes'])}")

asyncio.run(main())
