from huggingface_hub import hf_hub_download
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Dict, Optional, List, Union
import copy, uuid, requests, io, platform, pickle, os, urllib
from requests.adapters import Retry
from tqdm import tqdm


def _get_sep(path):
    if isinstance(path, bytes):
        return b'/'
    else:
        return '/'


def expanduser(path):
    """Expand ~ and ~user constructions.  If user or $HOME is unknown,
    do nothing."""
    path = os.fspath(path)
    if isinstance(path, bytes):
        tilde = b'~'
    else:
        tilde = '~'
    if not path.startswith(tilde):
        return path
    sep = _get_sep(path)
    i = path.find(sep, 1)
    if i < 0:
        i = len(path)
    if i == 1:
        if 'HOME' not in os.environ:
            import pwd
            try:
                userhome = pwd.getpwuid(os.getuid()).pw_dir
            except KeyError:
                # bpo-10496: if the current user identifier doesn't exist in the
                # password database, return the path unchanged
                return path
        else:
            userhome = os.environ['HOME']
    else:
        import pwd
        name = path[1:i]
        if isinstance(name, bytes):
            name = str(name, 'ASCII')
        try:
            pwent = pwd.getpwnam(name)
        except KeyError:
            # bpo-10496: if the user name from the path doesn't exist in the
            # password database, return the path unchanged
            return path
        userhome = pwent.pw_dir
    if isinstance(path, bytes):
        userhome = os.fsencode(userhome)
        root = b'/'
    else:
        root = '/'
    userhome = userhome.rstrip(root)
    return (userhome + path[i:]) or root



class ModelScopeConfig:
    DEFAULT_CREDENTIALS_PATH = Path.home().joinpath('.modelscope', 'credentials')
    path_credential = expanduser(DEFAULT_CREDENTIALS_PATH)
    COOKIES_FILE_NAME = 'cookies'
    GIT_TOKEN_FILE_NAME = 'git_token'
    USER_INFO_FILE_NAME = 'user'
    USER_SESSION_ID_FILE_NAME = 'session'

    @staticmethod
    def make_sure_credential_path_exist():
        os.makedirs(ModelScopeConfig.path_credential, exist_ok=True)

    @staticmethod
    def get_user_session_id():
        session_path = os.path.join(ModelScopeConfig.path_credential,
                                    ModelScopeConfig.USER_SESSION_ID_FILE_NAME)
        session_id = ''
        if os.path.exists(session_path):
            with open(session_path, 'rb') as f:
                session_id = str(f.readline().strip(), encoding='utf-8')
                return session_id
        if session_id == '' or len(session_id) != 32:
            session_id = str(uuid.uuid4().hex)
            ModelScopeConfig.make_sure_credential_path_exist()
            with open(session_path, 'w+') as wf:
                wf.write(session_id)

        return session_id

    @staticmethod
    def get_user_agent(user_agent: Union[Dict, str, None] = None, ) -> str:
        """Formats a user-agent string with basic info about a request.

        Args:
            user_agent (`str`, `dict`, *optional*):
                The user agent info in the form of a dictionary or a single string.

        Returns:
            The formatted user-agent string.
        """

        # include some more telemetrics when executing in dedicated
        # cloud containers
        MODELSCOPE_CLOUD_ENVIRONMENT = 'MODELSCOPE_ENVIRONMENT'
        MODELSCOPE_CLOUD_USERNAME = 'MODELSCOPE_USERNAME'
        env = 'custom'
        if MODELSCOPE_CLOUD_ENVIRONMENT in os.environ:
            env = os.environ[MODELSCOPE_CLOUD_ENVIRONMENT]
        user_name = 'unknown'
        if MODELSCOPE_CLOUD_USERNAME in os.environ:
            user_name = os.environ[MODELSCOPE_CLOUD_USERNAME]

        ua = 'modelscope/%s; python/%s; session_id/%s; platform/%s; processor/%s; env/%s; user/%s' % (
            "1.15.0",
            platform.python_version(),
            ModelScopeConfig.get_user_session_id(),
            platform.platform(),
            platform.processor(),
            env,
            user_name,
        )
        if isinstance(user_agent, dict):
            ua += '; ' + '; '.join(f'{k}/{v}' for k, v in user_agent.items())
        elif isinstance(user_agent, str):
            ua += '; ' + user_agent
        return ua
    
    @staticmethod
    def get_cookies():
        cookies_path = os.path.join(ModelScopeConfig.path_credential,
                                    ModelScopeConfig.COOKIES_FILE_NAME)
        if os.path.exists(cookies_path):
            with open(cookies_path, 'rb') as f:
                cookies = pickle.load(f)
                return cookies
        return None



def modelscope_http_get_model_file(
    url: str,
    local_dir: str,
    file_name: str,
    file_size: int,
    cookies: CookieJar,
    headers: Optional[Dict[str, str]] = None,
):
    """Download remote file, will retry 5 times before giving up on errors.

    Args:
        url(str):
            actual download url of the file
        local_dir(str):
            local directory where the downloaded file stores
        file_name(str):
            name of the file stored in `local_dir`
        file_size(int):
            The file size.
        cookies(CookieJar):
            cookies used to authentication the user, which is used for downloading private repos
        headers(Dict[str, str], optional):
            http headers to carry necessary info when requesting the remote file

    Raises:
        FileDownloadError: File download failed.

    """
    get_headers = {} if headers is None else copy.deepcopy(headers)
    get_headers['X-Request-ID'] = str(uuid.uuid4().hex)
    temp_file_path = os.path.join(local_dir, file_name)
    # retry sleep 0.5s, 1s, 2s, 4s
    retry = Retry(
        total=5,
        backoff_factor=1,
        allowed_methods=['GET'])
    while True:
        try:
            progress = tqdm(
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                total=file_size,
                initial=0,
                desc='Downloading',
            )
            partial_length = 0
            if os.path.exists(
                    temp_file_path):  # download partial, continue download
                with open(temp_file_path, 'rb') as f:
                    partial_length = f.seek(0, io.SEEK_END)
                    progress.update(partial_length)
            if partial_length > file_size:
                break
            get_headers['Range'] = 'bytes=%s-%s' % (partial_length,
                                                    file_size - 1)
            with open(temp_file_path, 'ab') as f:
                r = requests.get(
                    url,
                    stream=True,
                    headers=get_headers,
                    cookies=cookies,
                    timeout=60)
                r.raise_for_status()
                for chunk in r.iter_content(
                        chunk_size=1024 * 1024 * 1):
                    if chunk:  # filter out keep-alive new chunks
                        progress.update(len(chunk))
                        f.write(chunk)
            progress.close()
            break
        except (Exception) as e:  # no matter what happen, we will retry.
            retry = retry.increment('GET', url, error=e)
            retry.sleep()


def get_endpoint():
    MODELSCOPE_URL_SCHEME = 'https://'
    DEFAULT_MODELSCOPE_DOMAIN = 'www.modelscope.cn'
    modelscope_domain = os.getenv('MODELSCOPE_DOMAIN',
                                  DEFAULT_MODELSCOPE_DOMAIN)
    return MODELSCOPE_URL_SCHEME + modelscope_domain


def get_file_download_url(model_id: str, file_path: str, revision: str):
    """Format file download url according to `model_id`, `revision` and `file_path`.
    e.g., Given `model_id=john/bert`, `revision=master`, `file_path=README.md`,
    the resulted download url is: https://modelscope.cn/api/v1/models/john/bert/repo?Revision=master&FilePath=README.md

    Args:
        model_id (str): The model_id.
        file_path (str): File path
        revision (str): File revision.

    Returns:
        str: The file url.
    """
    file_path = urllib.parse.quote_plus(file_path)
    revision = urllib.parse.quote_plus(revision)
    download_url_template = '{endpoint}/api/v1/models/{model_id}/repo?Revision={revision}&FilePath={file_path}'
    return download_url_template.format(
        endpoint=get_endpoint(),
        model_id=model_id,
        revision=revision,
        file_path=file_path,
    )


def download_from_modelscope(model_id, origin_file_path, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    if os.path.basename(origin_file_path) in os.listdir(local_dir):
        print(f"{os.path.basename(origin_file_path)} has been already in {local_dir}.")
        return
    else:
        print(f"Start downloading {os.path.join(local_dir, os.path.basename(origin_file_path))}")
    headers = {'user-agent': ModelScopeConfig.get_user_agent(user_agent=None)}
    cookies = ModelScopeConfig.get_cookies()
    url = get_file_download_url(model_id=model_id, file_path=origin_file_path, revision="master")
    modelscope_http_get_model_file(
        url,
        local_dir,
        os.path.basename(origin_file_path),
        file_size=0,
        headers=headers,
        cookies=cookies
    )


def download_from_huggingface(model_id, origin_file_path, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    if os.path.basename(origin_file_path) in os.listdir(local_dir):
        print(f"{os.path.basename(origin_file_path)} has been already in {local_dir}.")
        return
    else:
        print(f"Start downloading {os.path.join(local_dir, os.path.basename(origin_file_path))}")
    hf_hub_download(model_id, origin_file_path, local_dir=local_dir)
