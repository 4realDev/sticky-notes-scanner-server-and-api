# pip install requests

import aiohttp
import nest_asyncio
import asyncio
import requests
import json
from aiohttp import FormData

nest_asyncio.apply()

# VARS FOR MIRO REST API
DEBUG_PRINT_RESPONSES = False
auth_token = "eyJtaXJvLm9yaWdpbiI6ImV1MDEifQ_dJFmha4uq9mmYZ9YP3jR42x9_98"
headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": f"Bearer {auth_token}"
}


# MIRO REST API HELPER FUNCTIONS
# GET ALL BOARD IDS AND NAMES
# WARNING:
# seems that the Get boards REST API function is not working properly
# sometimes it only returns one board, even if there are more


async def get_all_miro_board_names_and_ids(session):
    url = "https://api.miro.com/v2/boards?limit=50&sort=default"
    async with session.get(url, headers=headers) as resp:
        response = await resp.json()

        if DEBUG_PRINT_RESPONSES:
            print(await resp.text())
        return [{
            "name": board['name'],
            "id": board['id']
        } for board in response['data']]


# GET ALL BOARD ITEMS
# default limit is maximum (50)
async def get_all_items(
    board_id: str,
    session: aiohttp.client.ClientSession,
    max_num_of_items: int = 50,
    cursor: str = "",
    item_type="",
):
    url = f"https://api.miro.com/v2/boards/{board_id.replace('=', '%3D')}/items?limit={max_num_of_items}"

    if (item_type != ""):
        url = url + f"&type={item_type}"

    global response
    async with session.get(url, headers=headers) as resp:
        response = await resp.json()
        if DEBUG_PRINT_RESPONSES:
            print(await resp.text())

    # TODO: Find out how to solve this problem in python
    # if ('cursor' not in response):
    #     return response['data']

    # cursor = response['cursor']
    # new_url = url + f"&cursor={cursor}"

    # async with session.get(url, headers=headers) as resp:
    #     response = await resp.json()

    # cursor = response['cursor']
    # new_url = url + f"&cursor={cursor}"
    # print(new_url)
    # async with session.get(url, headers=headers) as resp:
    #     response = await resp.json()

    # cursor = response['cursor']
    # new_url = url + f"&cursor={cursor}"
    # print(new_url)
    # async with session.get(url, headers=headers) as resp:
    #     response = await resp.json()

    # while 'cursor' in response:
    #     cursor = response['cursor']
    #     new_url = url + f"&cursor={cursor}"
    #     print(new_url)
    #     async with session.get(url, headers=headers) as resp:
    #         new_response = await resp.json()
    #         print(new_response['cursor'])
    #         if DEBUG_PRINT_RESPONSES:
    #             print(await resp.text())
    #         response = new_response

    return response['data']


# CREATE FRAME
async def create_frame(
    pos_x: float,
    pos_y: float,
    title: str,
    height: float,
    width: float,
    board_id: str,
    session: aiohttp.client.ClientSession
):
    url = f"https://api.miro.com/v2/boards/{board_id.replace('=', '%3D')}/frames"
    payload = {
        "data": {
            "format": "custom",
            "title": title,
            "type": "freeform"
        },
        "style": {"fillColor": "#ffffffff"},
        "position": {
            "origin": "center",
            "x": pos_x,
            "y": pos_y
        },
        "geometry": {
            "height": height,
            "width": width
        }
    }

    headers = {
        "accept": "*/*",
        "content-type": "application/json",
        "authorization": f"Bearer {auth_token}"
    }

    async with session.post(url, json=payload, headers=headers) as resp:
        response = await resp.json()
        if DEBUG_PRINT_RESPONSES:
            print(await resp.text())
        if resp.status == requests.codes.created:
            print(f"Successfully created frame named {title}.")
        return response['id']


# DELETE FRAME
async def delete_frame(
    frame_id: str,
    board_id: str,
    session: aiohttp.client.ClientSession
):
    url = f"https://api.miro.com/v2/boards/{board_id.replace('=', '%3D')}/frames/{frame_id}"
    # response = requests.delete(url, headers=headers)
    # if DEBUG_PRINT_RESPONSES:
    #     print(await response.text())
    async with session.delete(url, headers=headers) as resp:
        if DEBUG_PRINT_RESPONSES:
            print(await resp.text())
        return resp.status


# CREATE ITEM
async def create_sticky_note(
    pos_x: float,
    pos_y: float,
    width: float,
    shape: str,
    color: str,
    text: str,
    board_id: str,
    parent_id: str,
    session: aiohttp.client.ClientSession
):
    url = f"https://api.miro.com/v2/boards/{board_id.replace('=', '%3D')}/sticky_notes"
    payload = {
        "data": {
            "content": f"<p>{text}</p>",
            "shape": shape
        },
        "style": {"fillColor": color},
        "position": {
            "origin": "center",
            "x": pos_x,
            "y": pos_y
        },
        "geometry": {
            "width": width
        },
        "parent": {"id": parent_id}
    }

    async with session.post(url=url, json=payload, headers=headers) as resp:
        response = await resp.json()
        if DEBUG_PRINT_RESPONSES:
            print(await resp.text())

        # if resp.status == requests.codes.created:
        #     print(
        #         f"Successfully created sticky note with with the text {text}.")

        return [resp.status, response["id"]]


# UPDATE STICKY NOTE
async def update_sticky_note(
    board_id: str,
    item_id: str,
    session: aiohttp.client.ClientSession
):
    url = f"https://api.miro.com/v2/boards/{board_id.replace('=', '%3D')}/sticky_notes/{item_id}"
    payload = {
        "style": {"fillColor": "red"},
    }

    async with session.patch(url=url, json=payload, headers=headers) as resp:
        response = await resp.json()
        # if DEBUG_PRINT_RESPONSES:
        print(await resp.text())

        # if resp.status == requests.codes.created:
        #     print(
        #         f"Successfully created sticky note with with the text {text}.")

        return resp.status


# CREATE ITEM WITH DATA
async def create_sticky_note_with_data(
    data,
    board_id: str,
    session: aiohttp.client.ClientSession
):
    url = f"https://api.miro.com/v2/boards/{board_id.replace('=', '%3D')}/sticky_notes"

    async with session.post(url=url, json=data, headers=headers) as resp:
        response = await resp.json()
        if DEBUG_PRINT_RESPONSES:
            print(await resp.text())

        # if resp.status == requests.codes.created:
        #     print(
        #         f"Successfully created sticky note with with the text {text}.")

        return resp.status


async def create_widget_with_data(
    data,
    widget_type,
    board_id: str,
    session: aiohttp.client.ClientSession,
):

    url = f"https://api.miro.com/v2/boards/{board_id.replace('=', '%3D')}/{widget_type}s"

    for element in list(data):
        if 'position' in element:
            for subElement in list(data['position']):
                if 'relativeTo' in subElement:
                    del data['position']['relativeTo']
        if 'geometry' in element:
            for subElement in list(data['geometry']):
                if widget_type == 'sticky_note' and 'height' in subElement:
                    del data['geometry']['height']
        # if 'parent' in element:
        #     for subElement in list(data['parent']):
        #         if 'links' in subElement:
        #             del data['parent']['links']
        if 'parent' in element:
            del data['parent']
        # This solves a bug on Miro -> Miro returns transparent elements with fillColor: "transparent" as
        # "style" : {"fillColor" : "#ffffff", "fillOpacity" : "0.0"}
        # setting the fillColor to "transparent" does not work ->
        # "message" : "Color value has invalid hex string, only css compatible hex strings are allowed"
        # the solution is to delete the fillColor property in this case
        # the element will be created then with: style: {fillColor: "transparent", fillOpacity: 1 ... } as it should be
        if 'style' in element:
            for subElement in list(data['style']):
                if 'fillColor' in subElement and \
                    'fillOpacity' in subElement and \
                    data['style']['fillColor'] == '#ffffff' and \
                        data['style']['fillOpacity'] == '0.0':
                    del data['style']['fillColor']
        if 'createdAt' in element:
            del data['createdAt']
        if 'createdAt' in element:
            del data['createdBy']
        if 'modifiedAt' in element:
            del data['modifiedAt']
        if 'modifiedBy' in element:
            del data['modifiedBy']
        if 'links' in element:
            del data['links']
        if 'id' in element:
            del data['id']
        if 'type' in element:
            del data['type']

    async with session.post(url=url, json=data, headers=headers) as resp:
        response = await resp.json()
        # if DEBUG_PRINT_RESPONSES:
        # print(await resp.text())

        # if resp.status == requests.codes.created:
        #     print(
        #         f"Successfully created sticky note with with the text {text}.")

        return resp.status


# CREATE LINE
async def create_line(
    pos_x: float,
    pos_y: float,
    width: float,
    height: float,
    color: str,
    board_id: str,
    parent_id: str,
    session: aiohttp.client.ClientSession
):
    url = f"https://api.miro.com/v2/boards/{board_id.replace('=', '%3D')}/shapes"

    payload = {
        "data": {"shape": "round_rectangle"},
        "style": {"fillColor": color},
        "position": {
            "origin": "center",
            "x": pos_x,
            "y": pos_y
        },
        "geometry": {
            "height": height,
            "width": width
        },
        "parent": {"id": parent_id}
    }

    async with session.post(url=url, json=payload, headers=headers) as resp:
        response = await resp.json()
        if DEBUG_PRINT_RESPONSES:
            print(await resp.text())

        return resp.status


# CREATE IMAGE
async def create_image(
    pos_x: float,
    pos_y: float,
    width: float,
    title: str,
    path: str,
    board_id: str,
    parent_id: str,
    session: aiohttp.client.ClientSession
):
    url = f"https://api.miro.com/v2/boards/{board_id.replace('=', '%3D')}/images"

    headers = {
        "accept": "*/*",
        "authorization": f"Bearer {auth_token}"
    }

    payload = {
        "title": title,
        "position": {
            "x": pos_x,
            "y": pos_y,
            "origin": "center"
        },
        "geometry": {
            "width": width,
            "rotation": 0
        },
        "parent": {"id": parent_id}
    }

    # fmt: off
    data = FormData()
    data.add_field('resource', open(path, "rb"), filename=f'{title}.png', content_type="application/png")
    data.add_field('data', json.dumps(payload), content_type="application/json")
    # fmt: on

    async with session.post(url=url,  data=data, headers=headers) as resp:
        response = await resp.json()
        if DEBUG_PRINT_RESPONSES:
            print(await resp.text())

        # if resp.status == requests.codes.created:
        #     print(
        #         f"Successfully created image of the sticky note with the path {path}.")

        return resp.status


# CREATE TEXT
async def create_text(
    pos_x: float,
    pos_y: float,
    width: float,
    fontSize: str,
    content: str,
    board_id: str,
    parent_id: str,
    session: aiohttp.client.ClientSession
):
    url = f"https://api.miro.com/v2/boards/{board_id.replace('=', '%3D')}/texts"

    payload = {
        "data": {"content": content},
        "style": {"fontSize": fontSize},
        "position": {
            "origin": "center",
            "x": pos_x,
            "y": pos_y
        },
        "geometry": {
            "width": width
        },
        "parent": {"id": parent_id}
    }

    async with session.post(url=url, json=payload, headers=headers) as resp:
        response = await resp.json()
        if DEBUG_PRINT_RESPONSES:
            print(await resp.text())

        return resp.status


# GET ALL TAGS FROM BOARD
async def get_tags_from_board(board_id: str, session: aiohttp.client.ClientSession):
    url = f"https://api.miro.com/v2/boards/{board_id.replace('=', '%3D')}/tags"
    async with session.get(url, headers=headers) as resp:
        response = await resp.json()

        if DEBUG_PRINT_RESPONSES:
            print(await resp.text())
        return [{
            "title": tag['title'],
            "id": tag['id']
        } for tag in response['data']]


# CREATE TAG
async def create_tag(
    title: str,
    fillColor: str,
    board_id: str,
    session: aiohttp.client.ClientSession
):
    url = f"https://api.miro.com/v2/boards/{board_id.replace('=', '%3D')}/tags"

    payload = {
        "fillColor": fillColor,
        "title": title
    }

    async with session.post(url=url, json=payload, headers=headers) as resp:
        response = await resp.json()
        if DEBUG_PRINT_RESPONSES:
            print(await resp.text())

        print(response)
        return response['id']


# ATTACH TAG TO ITEM
async def attach_tag_to_item(
    item_id: str,
    tag_id: str,
    board_id: str,
    session: aiohttp.client.ClientSession
):
    headersX = {
        "accept": "application/json",
        "authorization": "Bearer eyJtaXJvLm9yaWdpbiI6ImV1MDEifQ_dJFmha4uq9mmYZ9YP3jR42x9_98"
    }

    url = f"https://api.miro.com/v2/boards/{board_id.replace('=', '%3D')}/items/{item_id}?tag_id={tag_id}"
    # url = "https://api.miro.com/v2/boards/uXjVMLwQ46s%3D/items/3458764553635638305?tag_id=3458764553636201134"

    async with session.post(url=url, headers=headersX) as resp:
        # response = await resp.json()
        if DEBUG_PRINT_RESPONSES:
            print(await resp.text())

        # return response["id"]


# CREATE NEW MIRO-BOARD
# (if no Board with given name exists, else return the id or the existing one)
async def create_new_miro_board_or_get_existing(
    name: str,
    description: str,
    create_new_board: bool,
    session: aiohttp.client.ClientSession
) -> str:
    board_names_and_ids = await asyncio.create_task(get_all_miro_board_names_and_ids(session))
    print(f"\nExisting Boards: {board_names_and_ids}")

    # 1. save_in_existing_miro_board flag is set manually to "True" (default is "False")
    #    search for the given board name inside all existing miro boards
    if create_new_board == False:
        # 1.1 Board with the board given name exists -> return its id
        for board_name_and_id in board_names_and_ids:
            if board_name_and_id['name'] == name:
                print(
                    f"\nINFO: The board with the name {name} already exist. \n")
                return board_name_and_id['id']

        # 1.2 Board with the given board name does not exist -> return ERROR and stop function
        print(f"\n ERROR: The 'Create new Miro Board' checkbox is set to {create_new_board} and the given miro board name {name} was not found inside all miro boards. It could be possible that the searched board does not exist or must be still indexed from MIRO. Please wait a few seconds and try again or check the 'Create new Miro Board' checkbox to create a new miro board. \n")
        return "-1"

    # 2. save_in_existing_miro_board flag is "False" -> create a new miro board with the given name
    #    (no matter if the board with this name already exist)
    url = "https://api.miro.com/v2/boards"

    payload = {
        "name": name,
        "description": description,
        "policy": {
            "permissionsPolicy": {
                "collaborationToolsStartAccess": "all_editors",
                "copyAccess": "anyone",
                "sharingAccess": "team_members_with_editing_rights"
            },
            "sharingPolicy": {
                "access": "private",
                "inviteToAccountAndBoardLinkAccess": "no_access",
                "organizationAccess": "private",
                "teamAccess": "private"
            }
        }
    }

    async with session.post(url, json=payload, headers=headers) as resp:
        response = await resp.json()
        if DEBUG_PRINT_RESPONSES:
            print(await resp.text())

        if resp.status == requests.codes.created:
            print(
                f"Successfully created new miro board with the name {name} and the board_id {response['id']}.")

    return response['id']


# DELETE BOARD ITEM
async def delete_item(
    item_id: str,
    board_id: str,
    session: aiohttp.client.ClientSession
):

    url: str = f"https://api.miro.com/v2/boards/{board_id.replace('=', '%3D')}/items/{item_id}"

    # response = requests.delete(url, headers=headers)
    # if DEBUG_PRINT_RESPONSES:
    #     print(await response.text())

    async with session.delete(url, headers=headers) as resp:
        if DEBUG_PRINT_RESPONSES:
            print(await resp.text())
        return resp.status

    # headers = {
    #     "Accept": "application/json",
    #     "Authorization": "Bearer eyJtaXJvLm9yaWdpbiI6ImV1MDEifQ_Bw1UPxNElmWbBGy8MfSWWJWOCLs"
    # }

    # response = requests.delete(url, headers=headers)

    # # if DEBUG_PRINT_RESPONSES:
    # print(await resp.text())

    return response.status_code

    # WARNING: Seems not to work with global_session.delete(url, headers=headers)
    #     async with global_session.delete(url, headers=headers) as resp:
    #         response = await resp.json()
    #         if DEBUG_PRINT_RESPONSES: print(await resp.text())

# !!! NOT IN USE !!!
# DELETE ALL BOARD ITEMS


async def delete_all_items(
    board_id: str,
    session: aiohttp.client.ClientSession,
    item_type: str
):
    board_items = await asyncio.create_task(get_all_items(
        board_id=board_id,
        session=session,
        item_type=item_type
    ))
    for board_item in board_items:
        await asyncio.create_task(delete_item(board_item['id'], session=session))


# CREATE LIST OF ITEMS
async def create_all_sticky_notes(sticky_note_positions):
    global global_session
    global global_board_id

    url = f"https://api.miro.com/v2/boards/{global_board_id.replace('=', '%3D')}/sticky_notes"

    for sticky_note_position in sticky_note_positions:
        payload = {
            "data": {"shape": "square"},
            "position": {
                "origin": "center",
                "x": sticky_note_position['xmin'],
                "y": sticky_note_position['ymin']
            },
            "geometry": {
                #             "height": sticky_note_position['ymax'] - sticky_note_position['ymin'],
                "width": sticky_note_position['xmax'] - sticky_note_position['xmin']
            }
        }

        async with global_session.post(url, json=payload, headers=headers) as resp:
            response = await resp.json()
            if DEBUG_PRINT_RESPONSES:
                print(await resp.text())
