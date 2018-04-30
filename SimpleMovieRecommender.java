import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * User-user collaborative filtering movie recommender.
 */
public class SimpleMovieRecommender implements BaseMovieRecommender {

	//<mid>,<title field>,<tags> where the title field may be quoted and contain ','
	private static final Pattern MOVIE_LINE = Pattern.compile("^(\\d+),(.*),([^,]*)$");
	//<title> (<year>)
	private static final Pattern TITLE_YEAR = Pattern.compile("^(.*?)\\s*\\((\\d{4})\\)\\s*$");

	private Map<Integer, Movie> movies = new HashMap<Integer, Movie>();
	private Map<Integer, User> users = new HashMap<Integer, User>();

	//model information, filled by loadModel()
	private Map<Integer, Integer> userIndex = new HashMap<Integer, Integer>();				//uid -> matrix index
	private double[] meanRatings = new double[0];											//matrix index -> average rating
	private Map<Integer, Map<Integer, Double>> movieRaters = new HashMap<Integer, Map<Integer, Double>>();	//mid -> (user index -> rating)
	private double[][] userSim = new double[0][0];

	@Override
	public Map<Integer, Movie> loadMovies(String movieFilename) {
		if(movieFilename == null) return null;
		Map<Integer, Movie> result = new HashMap<Integer, Movie>();
		List<String> lines = readLines(movieFilename);

		//skip the header
		for(int i = 1; i < lines.size(); i++)
		{
			Matcher lineMatcher = MOVIE_LINE.matcher(lines.get(i).trim());
			if(!lineMatcher.matches()) continue;

			int mid = Integer.parseInt(lineMatcher.group(1));
			String title = lineMatcher.group(2).trim();
			if(title.length() >= 2 && title.startsWith("\"") && title.endsWith("\""))
			{
				title = title.substring(1, title.length() - 1);
			}

			//movies without a year keep their full title and year -1
			int year = -1;
			Matcher titleMatcher = TITLE_YEAR.matcher(title);
			if(titleMatcher.matches())
			{
				title = titleMatcher.group(1).trim();
				year = Integer.parseInt(titleMatcher.group(2));
			}

			Movie m = new Movie(mid, title.trim(), year);
			for(String tag: lineMatcher.group(3).split("\\|"))
			{
				if(!tag.trim().isEmpty()) m.addTag(tag.trim());
			}
			result.put(mid, m);
		}
		return result;
	}

	@Override
	public Map<Integer, User> loadUsers(String ratingFilename) {
		if(ratingFilename == null) return null;
		Map<Integer, User> result = new HashMap<Integer, User>();
		List<String> lines = readLines(ratingFilename);

		//skip the header
		for(int i = 1; i < lines.size(); i++)
		{
			String[] fields = lines.get(i).trim().split(",");
			if(fields.length != 4) continue;
			try
			{
				int uid = Integer.parseInt(fields[0].trim());
				int mid = Integer.parseInt(fields[1].trim());
				double rating = Double.parseDouble(fields[2].trim());
				long timestamp = Long.parseLong(fields[3].trim());

				//ratings of unknown movies cannot be stored in a Rating object
				Movie m = movies.get(mid);
				if(m == null) continue;

				User u = result.get(uid);
				if(u == null)
				{
					u = new User(uid);
					result.put(uid, u);
				}
				u.addRating(m, rating, timestamp);
			}
			catch(NumberFormatException e)
			{
				//skip malformed lines
			}
		}
		return result;
	}

	@Override
	public void loadData(String movieFilename, String userFilename) {
		Map<Integer, Movie> loadedMovies = loadMovies(movieFilename);
		movies = (loadedMovies != null)? loadedMovies : new HashMap<Integer, Movie>();
		Map<Integer, User> loadedUsers = loadUsers(userFilename);
		users = (loadedUsers != null)? loadedUsers : new HashMap<Integer, User>();
	}

	@Override
	public Map<Integer, Movie> getAllMovies() {
		return movies;
	}

	@Override
	public Map<Integer, User> getAllUsers() {
		return users;
	}

	@Override
	public void trainModel(String modelFilename) {
		//matrix indices follow ascending uid/mid
		List<Integer> uids = new ArrayList<Integer>(users.keySet());
		Collections.sort(uids);
		List<Integer> mids = new ArrayList<Integer>(movies.keySet());
		Collections.sort(mids);
		int numUsers = uids.size();
		int numMovies = mids.size();

		System.out.println("@@@ Computing user rating matrix");
		User[] indexedUsers = new User[numUsers];
		double[] means = new double[numUsers];
		for(int i = 0; i < numUsers; i++)
		{
			indexedUsers[i] = users.get(uids.get(i));
			means[i] = indexedUsers[i].getMeanRating();
		}

		System.out.println("@@@ Computing user sim matrix");
		double[][] sim = new double[numUsers][numUsers];
		for(int i = 0; i < numUsers; i++)
		{
			sim[i][i] = 1.0;
			for(int j = i + 1; j < numUsers; j++)
			{
				sim[i][j] = similarity(indexedUsers[i], means[i], indexedUsers[j], means[j]);
				sim[j][i] = sim[i][j];
			}
		}

		System.out.println("@@@ Writing out model file");
		Map<Integer, Integer> userMap = new TreeMap<Integer, Integer>();
		for(int i = 0; i < numUsers; i++) userMap.put(i, uids.get(i));
		Map<Integer, Integer> movieMap = new TreeMap<Integer, Integer>();
		for(int j = 0; j < numMovies; j++) movieMap.put(j, mids.get(j));

		try(BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(modelFilename), StandardCharsets.UTF_8)))
		{
			out.write("@NUM_USERS " + numUsers + "\n");
			out.write("@USER_MAP " + userMap + "\n");
			out.write("@NUM_MOVIES " + numMovies + "\n");
			out.write("@MOVIE_MAP " + movieMap + "\n");

			out.write("@RATING_MATRIX\n");
			StringBuilder row = new StringBuilder();
			for(int i = 0; i < numUsers; i++)
			{
				row.setLength(0);
				for(int j = 0; j < numMovies; j++)
				{
					Rating r = indexedUsers[i].ratings.get(mids.get(j));
					row.append((r != null)? r.rating : 0.0).append(' ');
				}
				row.append(means[i]).append(" \n");
				out.write(row.toString());
			}

			out.write("@USERSIM_MATRIX\n");
			for(int i = 0; i < numUsers; i++)
			{
				row.setLength(0);
				for(int j = 0; j < numUsers; j++)
				{
					row.append(sim[i][j]).append(' ');
				}
				row.append('\n');
				out.write(row.toString());
			}
		}
		catch(IOException e)
		{
			System.out.println("Error: cannot write model file " + modelFilename);
		}
	}

	@Override
	public void loadModel(String modelFilename) {
		userIndex = new HashMap<Integer, Integer>();
		movieRaters = new HashMap<Integer, Map<Integer, Double>>();
		List<String> lines = readLines(modelFilename);

		int numUsers = 0;
		int[] indexedMids = new int[0];
		int lineNum = 0;
		while(lineNum < lines.size())
		{
			String line = lines.get(lineNum++).trim();
			if(line.startsWith("@NUM_USERS"))
			{
				numUsers = Integer.parseInt(line.substring("@NUM_USERS".length()).trim());
			}
			else if(line.startsWith("@USER_MAP"))
			{
				for(Map.Entry<Integer, Integer> e: parseIndexMap(line.substring("@USER_MAP".length())).entrySet())
				{
					userIndex.put(e.getValue(), e.getKey());
				}
			}
			else if(line.startsWith("@NUM_MOVIES"))
			{
				indexedMids = new int[Integer.parseInt(line.substring("@NUM_MOVIES".length()).trim())];
			}
			else if(line.startsWith("@MOVIE_MAP"))
			{
				for(Map.Entry<Integer, Integer> e: parseIndexMap(line.substring("@MOVIE_MAP".length())).entrySet())
				{
					indexedMids[e.getKey()] = e.getValue();
				}
			}
			else if(line.startsWith("@RATING_MATRIX"))
			{
				//only non-zero ratings are kept, grouped by movie for fast lookup in predict()
				meanRatings = new double[numUsers];
				for(int i = 0; i < numUsers; i++)
				{
					String[] values = lines.get(lineNum++).trim().split(" ");
					for(int j = 0; j < indexedMids.length; j++)
					{
						if(values[j].equals("0.0")) continue;
						double rating = Double.parseDouble(values[j]);
						if(rating <= 0) continue;
						Map<Integer, Double> raters = movieRaters.get(indexedMids[j]);
						if(raters == null)
						{
							raters = new HashMap<Integer, Double>();
							movieRaters.put(indexedMids[j], raters);
						}
						raters.put(i, rating);
					}
					meanRatings[i] = Double.parseDouble(values[indexedMids.length]);
				}
			}
			else if(line.startsWith("@USERSIM_MATRIX"))
			{
				userSim = new double[numUsers][numUsers];
				for(int i = 0; i < numUsers; i++)
				{
					String[] values = lines.get(lineNum++).trim().split(" ");
					for(int j = 0; j < numUsers; j++)
					{
						userSim[i][j] = Double.parseDouble(values[j]);
					}
				}
			}
		}
	}

	@Override
	public double predict(Movie m, User u) {
		if(u == null) return 0;
		Integer ui = userIndex.get(u.uid);
		if(ui == null) return clamp(u.getMeanRating());
		double mean = meanRatings[ui];
		if(m == null) return clamp(mean);

		Map<Integer, Double> raters = movieRaters.get(m.mid);
		if(raters == null) return clamp(mean);

		//weighted sum of the other raters' deviations from their own average
		double numerator = 0;
		double denominator = 0;
		for(Map.Entry<Integer, Double> e: raters.entrySet())
		{
			int vi = e.getKey();
			if(vi == ui) continue;
			double s = userSim[ui][vi];
			numerator += s * (e.getValue() - meanRatings[vi]);
			denominator += Math.abs(s);
		}
		if(denominator == 0) return clamp(mean);
		return clamp(mean + numerator / denominator);
	}

	@Override
	public List<MovieItem> recommend(User u, int fromYear, int toYear, int K) {
		List<MovieItem> items = new ArrayList<MovieItem>();
		if(u == null || K <= 0) return items;

		for(Movie m: movies.values())
		{
			if(m.year >= fromYear && m.year <= toYear)
			{
				items.add(new MovieItem(m, predict(m, u)));
			}
		}
		Collections.sort(items);
		if(items.size() > K) return new ArrayList<MovieItem>(items.subList(0, K));
		return items;
	}

	/**
	 * Computes the similarity s(u, v) from the ratings of the movies both users rated,
	 * centered on each user's average rating. Returns 0 if the denominator is 0.
	 */
	private static double similarity(User u, double meanU, User v, double meanV)
	{
		//iterate over the user with fewer ratings
		User small = u, large = v;
		double meanSmall = meanU, meanLarge = meanV;
		if(u.ratings.size() > v.ratings.size())
		{
			small = v; large = u;
			meanSmall = meanV; meanLarge = meanU;
		}

		double numerator = 0, sumSqSmall = 0, sumSqLarge = 0;
		for(Map.Entry<Integer, Rating> e: small.ratings.entrySet())
		{
			Rating other = large.ratings.get(e.getKey());
			if(other == null) continue;
			double dSmall = e.getValue().rating - meanSmall;
			double dLarge = other.rating - meanLarge;
			numerator += dSmall * dLarge;
			sumSqSmall += dSmall * dSmall;
			sumSqLarge += dLarge * dLarge;
		}
		double denominator = Math.sqrt(sumSqSmall) * Math.sqrt(sumSqLarge);
		if(denominator == 0) return 0;
		double s = numerator / denominator;
		return Double.isFinite(s)? s : 0;
	}

	/**
	 * Parses "{0=<id1>, 1=<id2>, ...}" into a map from matrix index to id.
	 */
	private static Map<Integer, Integer> parseIndexMap(String text)
	{
		Map<Integer, Integer> result = new HashMap<Integer, Integer>();
		String body = text.trim();
		if(body.startsWith("{")) body = body.substring(1);
		if(body.endsWith("}")) body = body.substring(0, body.length() - 1);
		for(String entry: body.split(","))
		{
			String[] kv = entry.trim().split("=");
			if(kv.length != 2) continue;
			result.put(Integer.parseInt(kv[0].trim()), Integer.parseInt(kv[1].trim()));
		}
		return result;
	}

	/**
	 * Clamps a predicted rating to [0, 5]. NaN/Inf become 0.
	 */
	private static double clamp(double rating)
	{
		if(!Double.isFinite(rating)) return 0;
		return Math.max(0, Math.min(5.0, rating));
	}

	/**
	 * Reads all lines of a UTF-8 text file. Returns an empty list if the file cannot be read.
	 */
	private static List<String> readLines(String filename)
	{
		List<String> lines = new ArrayList<String>();
		if(filename == null || !new File(filename).isFile())
		{
			System.out.println("Error: cannot read file " + filename);
			return lines;
		}
		try(BufferedReader in = new BufferedReader(new InputStreamReader(new FileInputStream(filename), StandardCharsets.UTF_8)))
		{
			String line;
			while((line = in.readLine()) != null) lines.add(line);
		}
		catch(IOException e)
		{
			System.out.println("Error: cannot read file " + filename);
		}
		return lines;
	}

}
